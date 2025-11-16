"""
Child of the autopilot that additionally runs data collection and storage.
"""

import cv2
import carla
import random
import torch
import numpy as np
import json
import os
from enum import Enum
import gzip
import laspy
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from pathlib import Path
from common.utils import conv_NuScenes2Carla

from autopilot import AutoPilot
import transfuser_utils as t_u

from birds_eye_view.chauffeurnet import ObsManager
from birds_eye_view.run_stop_sign import RunStopSign
from leaderboard.autoagents import autonomous_agent, autonomous_agent_local
from PIL import Image

from agents.tools.misc import (is_within_distance, get_trafficlight_trigger_location, compute_distance)

from agents.navigation.local_planner import LocalPlanner

from common.config import GlobalConfig
from kinematic_bicycle_model import KinematicBicycleModel
from lateral_controller import LateralPIDController

def get_entry_point():
  return 'DataAgent'


class DataAgent(AutoPilot):
  """
        Child of the autopilot that additionally runs data collection and storage.
        """

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    #-------configuration for data agent----------------------
    self.conf_enable_disturbance = True #Add steering disturbance for agent
    self.conf_plot_control = True #Plot control signals during runtime
    #---------------------------------------------------------
    super().setup(path_to_conf_file, route_index, traffic_manager=None)
    #--------overwrite config----------------
    self.config = GlobalConfig()
    self.ego_model = KinematicBicycleModel(self.config)
    self.vehicle_model = KinematicBicycleModel(self.config)
    self._turn_controller = LateralPIDController(self.config)
    #----------------------------------------
    self.weather_tmp = None
    self.step_tmp = 0

    self.tm = traffic_manager

    self.scenario_name = Path(path_to_conf_file).parent.name
    self.cutin_vehicle_starting_position = None

    if self.save_path is not None and self.datagen:
      if self.config.gen_lidar:
        (self.save_path / 'lidar').mkdir()
      if self.config.gen_rgb:
        (self.save_path / 'rgb').mkdir()
        (self.save_path / 'rgb_augmented').mkdir()
        for idx, camera in enumerate(self.config.cameras):
          (self.save_path / 'rgb' / camera.name).mkdir()
      if self.config.gen_semantics:
        (self.save_path / 'semantics').mkdir()
        (self.save_path / 'semantics_augmented').mkdir()
      if self.config.gen_depth:
        (self.save_path / 'depth').mkdir()
        (self.save_path / 'depth_augmented').mkdir()
      if self.config.gen_bev_semantics:
        (self.save_path / 'bev_semantics').mkdir()
        (self.save_path / 'bev_semantics_augmented').mkdir()
      if self.config.gen_boxes:
        (self.save_path / 'boxes').mkdir()

    self.tmp_visu = int(os.environ.get('TMP_VISU', 0))

    self._active_traffic_light = None
    self.last_lidar = None
    self.last_ego_transform = None

    if self.conf_enable_disturbance:
      self.disturbance_generater = Disturbance_generater(rate_per_sec = 1, max_duration = 2, max_magnitude = 0.5, step_size = 1/20)

    if self.conf_plot_control:
      self.timestamp_10s = []
      self.steer_raw_10s = []
      self.steer_dist_10s = []
      self.steer_out_10s = []
      self.Throttle_10s = []
      self.brake_10s = []
      self.speed_10s = []

    self.step_count = 0

    #-----------over write parent parameters----------------------
    self.track = autonomous_agent.Track.MAP_QUALIFIER
    if self.conf_enable_disturbance:
      #Reduce responce to disturbance
      self._turn_controller.lateral_pid_kd = 0
    

    #extend sensor timeout because of machine resources
    self.sensor_interface._queue_timeout = 100

  def _init(self, hd_map):
    super()._init(hd_map)
    if self.datagen:
      self.shuffle_weather()

    obs_config = {
        'width_in_pixels': self.config.lidar_resolution_width,
        'pixels_ev_to_bottom': self.config.lidar_resolution_height / 2.0,
        'pixels_per_meter': self.config.pixels_per_meter_collection,
        'history_idx': [-1],
        'scale_bbox': True,
        'scale_mask_col': 1.0,
        'map_folder': 'maps_2ppm_cv'
    }

    self.stop_sign_criteria = RunStopSign(self._world)
    self.ss_bev_manager = ObsManager(obs_config, self.config)
    self.ss_bev_manager.attach_ego_vehicle(self._vehicle, criteria_stop=self.stop_sign_criteria)

    self.ss_bev_manager_augmented = ObsManager(obs_config, self.config)

    bb_copy = carla.BoundingBox(self._vehicle.bounding_box.location, self._vehicle.bounding_box.extent)
    transform_copy = carla.Transform(self._vehicle.get_transform().location, self._vehicle.get_transform().rotation)
    # Can't clone the carla vehicle object, so I use a dummy class with similar attributes.
    self.augmented_vehicle_dummy = t_u.CarlaActorDummy(self._vehicle.get_world(), bb_copy, transform_copy,
                                                       self._vehicle.id)
    self.ss_bev_manager_augmented.attach_ego_vehicle(self.augmented_vehicle_dummy,
                                                     criteria_stop=self.stop_sign_criteria)

    self._local_planner = LocalPlanner(self._vehicle, opt_dict={}, map_inst=self.world_map)

  def sensors(self):
    # workaraound that only does data augmentation at the beginning of the route
    if self.config.augment:
      self.augmentation_translation = np.random.uniform(low=self.config.camera_translation_augmentation_min,
                                                        high=self.config.camera_translation_augmentation_max)
      self.augmentation_rotation = np.random.uniform(low=self.config.camera_rotation_augmentation_min,
                                                     high=self.config.camera_rotation_augmentation_max)

    result = super().sensors()

    if self.save_path is not None and (self.datagen or self.tmp_visu):
      if self.config.gen_rgb:
        for idx, camera in enumerate(self.config.cameras):
          tmp_trans, tmp_rot = conv_NuScenes2Carla(camera.trans, camera.get_rot_mat())
          result.append({
          'type': 'sensor.camera.rgb',
          'x': tmp_trans[0],
          'y': tmp_trans[1],
          'z': tmp_trans[2],
          'roll': np.degrees(tmp_rot[0]),
          'pitch': np.degrees(tmp_rot[1]),
          'yaw': np.degrees(tmp_rot[2]),
          'width': camera.W,
          'height': camera.H,
          'fov': np.degrees(camera.fov),
          'id': camera.name
          })
          if camera.name == 'CAM_FRONT':
            result.append({
            'type': 'sensor.camera.rgb',
            'x': camera.trans[0],
            'y': camera.trans[1],
            'z': camera.trans[2],
            'roll': np.degrees(camera.rot_euler[0]),
            'pitch': np.degrees(camera.rot_euler[1]),
            'yaw': np.degrees(camera.rot_euler[2]),
            'width': camera.W,
            'height': camera.H,
            'fov': np.degrees(camera.fov),
            'id': 'rgb_augmented'
            })

      if self.config.gen_depth:
        result.append({
          'type': 'sensor.camera.depth',
          'x': self.config.camera_pos[0],
          'y': self.config.camera_pos[1],
          'z': self.config.camera_pos[2],
          'roll': self.config.camera_rot_0[0],
          'pitch': self.config.camera_rot_0[1],
          'yaw': self.config.camera_rot_0[2],
          'width': self.config.camera_width,
          'height': self.config.camera_height,
          'fov': self.config.camera_fov,
          'id': 'depth'
        })
        result.append({
          'type': 'sensor.camera.depth',
          'x': self.config.camera_pos[0],
          'y': self.config.camera_pos[1] + self.augmentation_translation,
          'z': self.config.camera_pos[2],
          'roll': self.config.camera_rot_0[0],
          'pitch': self.config.camera_rot_0[1],
          'yaw': self.config.camera_rot_0[2] + self.augmentation_rotation,
          'width': self.config.camera_width,
          'height': self.config.camera_height,
          'fov': self.config.camera_fov,
          'id': 'depth_augmented'
        })

      if self.config.gen_lidar:
        result.append({
          'type': 'sensor.lidar.ray_cast',
          'x': self.config.lidar_pos[0],
          'y': self.config.lidar_pos[1],
          'z': self.config.lidar_pos[2],
          'roll': self.config.lidar_rot[0],
          'pitch': self.config.lidar_rot[1],
          'yaw': self.config.lidar_rot[2],
          'rotation_frequency': self.config.lidar_rotation_frequency,
          'points_per_second': self.config.lidar_points_per_second,
          'id': 'lidar'
        })

      if self.config.gen_semantics:
        result.append({
          'type': 'sensor.camera.semantic_segmentation',
          'x': self.config.camera_pos[0],
          'y': self.config.camera_pos[1],
          'z': self.config.camera_pos[2],
          'roll': self.config.camera_rot_0[0],
          'pitch': self.config.camera_rot_0[1],
          'yaw': self.config.camera_rot_0[2],
          'width': self.config.camera_width,
          'height': self.config.camera_height,
          'fov': self.config.camera_fov,
          'id': 'semantics'
        })
        result.append({
          'type': 'sensor.camera.semantic_segmentation',
          'x': self.config.camera_pos[0],
          'y': self.config.camera_pos[1] + self.augmentation_translation,
          'z': self.config.camera_pos[2],
          'roll': self.config.camera_rot_0[0],
          'pitch': self.config.camera_rot_0[1],
          'yaw': self.config.camera_rot_0[2] + self.augmentation_rotation,
          'width': self.config.camera_width,
          'height': self.config.camera_height,
          'fov': self.config.camera_fov,
          'id': 'semantics_augmented'
        })

    return result

  def tick(self, input_data):
    result = {}

    if self.save_path is not None and (self.datagen or self.tmp_visu):
      if self.config.gen_rgb:
        rgb = {}
        for idx, camera in enumerate(self.config.cameras):
          rgb[camera.name] = input_data[camera.name][1][:, :, :3]
        rgb_augmented = input_data['rgb_augmented'][1][:, :, :3]
      else:
        rgb = None
        rgb_augmented = None

      if self.config.gen_depth:
        # We store depth at 8 bit to reduce the filesize. 16 bit would be ideal, but we can't afford the extra storage.
        depth = input_data['depth'][1][:, :, :3]
        depth = (t_u.convert_depth(depth) * 255.0 + 0.5).astype(np.uint8)
      
        depth_augmented = input_data['depth_augmented'][1][:, :, :3]
        depth_augmented = (t_u.convert_depth(depth_augmented) * 255.0 + 0.5).astype(np.uint8)
      else:
        depth = None
        depth_augmented = None

      if self.config.gen_semantics:
        semantics = input_data['semantics'][1][:, :, 2]
        semantics_augmented = input_data['semantics_augmented'][1][:, :, 2]
      else:
        semantics = None
        semantics_augmented = None

    else:
      rgb = None
      rgb_augmented = None
      semantics = None
      semantics_augmented = None
      depth = None
      depth_augmented = None

    # The 10 Hz LiDAR only delivers half a sweep each time step at 20 Hz.
    # Here we combine the 2 sweeps into the same coordinate system
    if self.last_lidar is not None:
      if self.config.gen_lidar:
        ego_transform = self._vehicle.get_transform()
        ego_location = ego_transform.location
        last_ego_location = self.last_ego_transform.location
        relative_translation = np.array([
            ego_location.x - last_ego_location.x, ego_location.y - last_ego_location.y,
            ego_location.z - last_ego_location.z
        ])

        ego_yaw = ego_transform.rotation.yaw
        last_ego_yaw = self.last_ego_transform.rotation.yaw
        relative_rotation = np.deg2rad(t_u.normalize_angle_degree(ego_yaw - last_ego_yaw))

        orientation_target = np.deg2rad(ego_yaw)
        # Rotate difference vector from global to local coordinate system.
        rotation_matrix = np.array([[np.cos(orientation_target), -np.sin(orientation_target), 0.0],
                                    [np.sin(orientation_target),
                                    np.cos(orientation_target), 0.0], [0.0, 0.0, 1.0]])
        relative_translation = rotation_matrix.T @ relative_translation

        lidar_last = t_u.algin_lidar(self.last_lidar, relative_translation, relative_rotation)
        # Combine back and front half of LiDAR
        lidar_360 = np.concatenate((input_data['lidar'], lidar_last), axis=0)
      else:
        lidar_360 = input_data['lidar']  # The first frame only has 1 half

    if self.config.gen_boxes:
      bounding_boxes = self.get_bounding_boxes(lidar=lidar_360)

    if self.config.gen_bev_semantics:
      self.stop_sign_criteria.tick(self._vehicle)
      bev_semantics = self.ss_bev_manager.get_observation(self.close_traffic_lights)
      bev_semantics_augmented = self.ss_bev_manager_augmented.get_observation(self.close_traffic_lights)

    if self.tmp_visu and self.config.gen_boxes and self.config.gen_rgb:
      self.visualuize(bev_semantics['rendered'], rgb['rgb_Front'])

    if self.config.gen_lidar:
      result['lidar'] = lidar_360
    if self.config.gen_rgb:
      result['rgb'] = rgb
      result['rgb_augmented'] = rgb_augmented
    if self.config.gen_semantics:
      result['semantics'] = semantics
      result['semantics_augmented'] = semantics_augmented
    if self.config.gen_depth:
      result['depth'] = depth
      result['depth_augmented'] = depth_augmented
    if self.config.gen_bev_semantics:
      result['bev_semantics'] = bev_semantics['bev_semantic_classes']
      result['bev_semantics_augmented'] = bev_semantics_augmented['bev_semantic_classes']
    if self.config.gen_boxes:
      result['bounding_boxes'] = bounding_boxes

    return result

  @torch.inference_mode()
  def run_step(self, input_data, timestamp, sensors=None, plant=False):
    self.step_tmp += 1

    if self.config.gen_lidar:
      # Convert LiDAR into the coordinate frame of the ego vehicle
      input_data['lidar'] = t_u.lidar_to_ego_coordinate(self.config, input_data['lidar'])

    # Must be called before run_step, so that the correct augmentation shift is saved
    if self.datagen:
      self.augment_camera(sensors)

    control = super().run_step(input_data, timestamp, plant=plant)

    tick_data = self.tick(input_data)

    if self.step % self.config.data_save_freq == 0:
      if self.save_path is not None and self.datagen:
        self.save_sensors(tick_data)

    if self.config.gen_lidar:
      self.last_lidar = input_data['lidar']
      self.last_ego_transform = self._vehicle.get_transform()

    if self.conf_enable_disturbance:
      self.add_disturbance(control, timestamp, input_data)

    if self.conf_plot_control:
      self.plot_status(input_data)

    self.step_count += 1

    if plant:
      # Control contains data when run with plant
      return {**tick_data, **control}
    else:
      return control

  def augment_camera(self, sensors):
    # Update dummy vehicle
    if self.initialized:
      # We are still rendering the map for the current frame, so we need to use the translation from the last frame.
      last_translation = self.augmentation_translation
      last_rotation = self.augmentation_rotation
      bb_copy = carla.BoundingBox(self._vehicle.bounding_box.location, self._vehicle.bounding_box.extent)
      transform_copy = carla.Transform(self._vehicle.get_transform().location, self._vehicle.get_transform().rotation)
      augmented_loc = transform_copy.transform(carla.Location(0.0, last_translation, 0.0))
      transform_copy.location = augmented_loc
      transform_copy.rotation.yaw = transform_copy.rotation.yaw + last_rotation
      self.augmented_vehicle_dummy.bounding_box = bb_copy
      self.augmented_vehicle_dummy.transform = transform_copy

  def _get_night_mode(self, weather):
    """Check wheather or not the street lights need to be turned on"""
    SUN_ALTITUDE_THRESHOLD_1 = 15
    SUN_ALTITUDE_THRESHOLD_2 = 165

    # For higher fog and cloudness values, the amount of light in scene starts to rapidly decrease
    CLOUDINESS_THRESHOLD = 80
    FOG_THRESHOLD = 40

    # In cases where more than one weather conditition is active, decrease the thresholds
    COMBINED_THRESHOLD = 10

    altitude_dist = weather.sun_altitude_angle - SUN_ALTITUDE_THRESHOLD_1
    altitude_dist = min(altitude_dist, SUN_ALTITUDE_THRESHOLD_2 - weather.sun_altitude_angle)
    cloudiness_dist = CLOUDINESS_THRESHOLD - weather.cloudiness
    fog_density_dist = FOG_THRESHOLD - weather.fog_density

    # Check each parameter independetly
    if altitude_dist < 0 or cloudiness_dist < 0 or fog_density_dist < 0:
      return True

    # Check if two or more values are close to their threshold
    joined_threshold = int(altitude_dist < COMBINED_THRESHOLD)
    joined_threshold += int(cloudiness_dist < COMBINED_THRESHOLD)
    joined_threshold += int(fog_density_dist < COMBINED_THRESHOLD)

    if joined_threshold >= 2:
      return True

    return False

  def shuffle_weather(self):
    # change weather for visual diversity
    if self.weather_tmp is None:
      t = carla.WeatherParameters
      options = dir(t)[:22]
      chosen_preset = random.choice(options)
      self.chosen_preset = chosen_preset
      weather = t.__getattribute__(t, chosen_preset)
      self.weather_tmp = weather

    self._world.set_weather(self.weather_tmp)

    # night mode
    vehicles = self._world.get_actors().filter('*vehicle*')
    if self._get_night_mode(weather):
      for vehicle in vehicles:
        vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))
    else:
      for vehicle in vehicles:
        vehicle.set_light_state(carla.VehicleLightState.NONE)

  def save_sensors(self, tick_data):
    frame = self.step // self.config.data_save_freq

    # CARLA images are already in opencv's BGR format.
    if self.config.gen_rgb:
      rgb = tick_data['rgb']
      for idx, camera in enumerate(self.config.cameras):
        cv2.imwrite(str(self.save_path / 'rgb' / camera.name /(f'{frame:04}.jpg')), rgb[camera.name])
      cv2.imwrite(str(self.save_path / 'rgb_augmented' / (f'{frame:04}.jpg')), tick_data['rgb_augmented'])

    if self.config.gen_semantics:
      cv2.imwrite(str(self.save_path / 'semantics' / (f'{frame:04}.png')), tick_data['semantics'])
      cv2.imwrite(str(self.save_path / 'semantics_augmented' / (f'{frame:04}.png')), tick_data['semantics_augmented'])

    if self.config.gen_depth:
      cv2.imwrite(str(self.save_path / 'depth' / (f'{frame:04}.png')), tick_data['depth'])
      cv2.imwrite(str(self.save_path / 'depth_augmented' / (f'{frame:04}.png')), tick_data['depth_augmented'])

    if self.config.gen_bev_semantics:
      cv2.imwrite(str(self.save_path / 'bev_semantics' / (f'{frame:04}.png')), tick_data['bev_semantics'])
      cv2.imwrite(str(self.save_path / 'bev_semantics_augmented' / (f'{frame:04}.png')),
                  tick_data['bev_semantics_augmented'])

    # Specialized LiDAR compression format
    if self.config.gen_lidar:
      header = laspy.LasHeader(point_format=self.config.point_format)
      header.offsets = np.min(tick_data['lidar'], axis=0)
      header.scales = np.array([self.config.point_precision, self.config.point_precision, self.config.point_precision])

      with laspy.open(self.save_path / 'lidar' / (f'{frame:04}.laz'), mode='w', header=header) as writer:
        point_record = laspy.ScaleAwarePointRecord.zeros(tick_data['lidar'].shape[0], header=header)
        point_record.x = tick_data['lidar'][:, 0]
        point_record.y = tick_data['lidar'][:, 1]
        point_record.z = tick_data['lidar'][:, 2]

        writer.write_points(point_record)

    if self.config.gen_boxes:
      with gzip.open(self.save_path / 'boxes' / (f'{frame:04}.json.gz'), 'wt', encoding='utf-8') as f:
        json.dump(tick_data['bounding_boxes'], f, indent=4)

  def destroy(self, results=None):
    torch.cuda.empty_cache()

    if results is not None and self.save_path is not None:
      with gzip.open(os.path.join(self.save_path, 'results.json.gz'), 'wt', encoding='utf-8') as f:
        json.dump(results.__dict__, f, indent=2)

    super().destroy(results)

  def get_bounding_boxes(self, lidar=None):
    results = []

    ego_transform = self._vehicle.get_transform()
    ego_control = self._vehicle.get_control()
    ego_velocity = self._vehicle.get_velocity()
    ego_matrix = np.array(ego_transform.get_matrix())
    ego_rotation = ego_transform.rotation
    ego_extent = self._vehicle.bounding_box.extent
    ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity)
    ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z])
    ego_yaw = np.deg2rad(ego_rotation.yaw)
    ego_brake = ego_control.brake

    relative_yaw = 0.0
    relative_pos = t_u.get_relative_transform(ego_matrix, ego_matrix)

    # Check for possible vehicle obstacles
    # Retrieve all relevant actors
    self._actors = self._world.get_actors()
    vehicle_list = self._actors.filter('*vehicle*')

    result = {
        'class': 'ego_car',
        'extent': [ego_dx[0], ego_dx[1], ego_dx[2]],
        'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
        'yaw': relative_yaw,
        'num_points': -1,
        'distance': -1,
        'speed': ego_speed,
        'brake': ego_brake,
        'id': int(self._vehicle.id),
        'matrix': ego_transform.get_matrix()
    }
    results.append(result)

    for vehicle in vehicle_list:
      if vehicle.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
        if vehicle.id != self._vehicle.id:
          vehicle_transform = vehicle.get_transform()
          vehicle_rotation = vehicle_transform.rotation
          vehicle_matrix = np.array(vehicle_transform.get_matrix())
          vehicle_control = vehicle.get_control()
          vehicle_velocity = vehicle.get_velocity()
          vehicle_extent = vehicle.bounding_box.extent
          vehicle_id = vehicle.id

          vehicle_extent_list = [vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]
          yaw = np.deg2rad(vehicle_rotation.yaw)

          relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
          relative_pos = t_u.get_relative_transform(ego_matrix, vehicle_matrix)
          vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity)
          vehicle_brake = vehicle_control.brake
          vehicle_steer = vehicle_control.steer
          vehicle_throttle = vehicle_control.throttle

          # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
          if not lidar is None:
            num_in_bbox_points = self.get_points_in_bbox(relative_pos, relative_yaw, vehicle_extent_list, lidar)
          else:
            num_in_bbox_points = -1

          distance = np.linalg.norm(relative_pos)

          result = {
              'class': 'car',
              'extent': vehicle_extent_list,
              'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
              'yaw': relative_yaw,
              'num_points': int(num_in_bbox_points),
              'distance': distance,
              'speed': vehicle_speed,
              'brake': vehicle_brake,
              'steer': vehicle_steer,
              'throttle': vehicle_throttle,
              'id': int(vehicle_id),
              'role_name': vehicle.attributes['role_name'],
              'type_id': vehicle.type_id,
              'matrix': vehicle_transform.get_matrix()
          }
          results.append(result)

    walkers = self._actors.filter('*walker*')
    for walker in walkers:
      if walker.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
        walker_transform = walker.get_transform()
        walker_velocity = walker.get_velocity()
        walker_rotation = walker.get_transform().rotation
        walker_matrix = np.array(walker_transform.get_matrix())
        walker_id = walker.id
        walker_extent = walker.bounding_box.extent
        walker_extent = [walker_extent.x, walker_extent.y, walker_extent.z]
        yaw = np.deg2rad(walker_rotation.yaw)

        relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
        relative_pos = t_u.get_relative_transform(ego_matrix, walker_matrix)

        walker_speed = self._get_forward_speed(transform=walker_transform, velocity=walker_velocity)

        # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
        if not lidar is None:
          num_in_bbox_points = self.get_points_in_bbox(relative_pos, relative_yaw, walker_extent, lidar)
        else:
          num_in_bbox_points = -1

        distance = np.linalg.norm(relative_pos)

        result = {
            'class': 'walker',
            'extent': walker_extent,
            'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
            'yaw': relative_yaw,
            'num_points': int(num_in_bbox_points),
            'distance': distance,
            'speed': walker_speed,
            'id': int(walker_id),
            'matrix': walker_transform.get_matrix()
        }
        results.append(result)

    # Note this only saves static actors, which does not include static background objects
    static_list = self._actors.filter('*static*')
    for static in static_list:
      if static.get_location().distance(self._vehicle.get_location()) < self.config.bb_save_radius:
        static_transform = static.get_transform()
        static_velocity = static.get_velocity()
        static_rotation = static.get_transform().rotation
        static_matrix = np.array(static_transform.get_matrix())
        static_id = static.id
        static_extent = static.bounding_box.extent
        static_extent = [static_extent.x, static_extent.y, static_extent.z]
        yaw = np.deg2rad(static_rotation.yaw)

        relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
        relative_pos = t_u.get_relative_transform(ego_matrix, static_matrix)

        static_speed = self._get_forward_speed(transform=static_transform, velocity=static_velocity)

        # Computes how many LiDAR hits are on a bounding box. Used to filter invisible boxes during data loading.
        if not lidar is None:
          num_in_bbox_points = self.get_points_in_bbox(relative_pos, relative_yaw, static_extent, lidar)
        else:
          num_in_bbox_points = -1

        distance = np.linalg.norm(relative_pos)

        result = {
            'class': 'static',
            'extent': static_extent,
            'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
            'yaw': relative_yaw,
            'num_points': int(num_in_bbox_points),
            'distance': distance,
            'speed': static_speed,
            'id': int(static_id),
            'matrix': static_transform.get_matrix(),
            'type_id': static.type_id,
            'mesh_path': static.attributes['mesh_path'] if 'mesh_path' in static.attributes else None
        }
        results.append(result)

    for traffic_light in self.close_traffic_lights:
      traffic_light_extent = [traffic_light[0].extent.x, traffic_light[0].extent.y, traffic_light[0].extent.z]

      traffic_light_transform = carla.Transform(traffic_light[0].location, traffic_light[0].rotation)
      traffic_light_rotation = traffic_light_transform.rotation
      traffic_light_matrix = np.array(traffic_light_transform.get_matrix())
      yaw = np.deg2rad(traffic_light_rotation.yaw)

      relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
      relative_pos = t_u.get_relative_transform(ego_matrix, traffic_light_matrix)

      distance = np.linalg.norm(relative_pos)

      result = {
          'class': 'traffic_light',
          'extent': traffic_light_extent,
          'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
          'yaw': relative_yaw,
          'distance': distance,
          'state': str(traffic_light[1]),
          'id': int(traffic_light[2]),
          'affects_ego': traffic_light[3],
          'matrix': traffic_light_transform.get_matrix()
      }
      results.append(result)

    for stop_sign in self.close_stop_signs:
      stop_sign_extent = [stop_sign[0].extent.x, stop_sign[0].extent.y, stop_sign[0].extent.z]

      stop_sign_transform = carla.Transform(stop_sign[0].location, stop_sign[0].rotation)
      stop_sign_rotation = stop_sign_transform.rotation
      stop_sign_matrix = np.array(stop_sign_transform.get_matrix())
      yaw = np.deg2rad(stop_sign_rotation.yaw)

      relative_yaw = t_u.normalize_angle(yaw - ego_yaw)
      relative_pos = t_u.get_relative_transform(ego_matrix, stop_sign_matrix)

      distance = np.linalg.norm(relative_pos)

      result = {
          'class': 'stop_sign',
          'extent': stop_sign_extent,
          'position': [relative_pos[0], relative_pos[1], relative_pos[2]],
          'yaw': relative_yaw,
          'distance': distance,
          'id': int(stop_sign[1]),
          'affects_ego': stop_sign[2],
          'matrix': stop_sign_transform.get_matrix()
      }
      results.append(result)

    return results

  def get_points_in_bbox(self, vehicle_pos, vehicle_yaw, extent, lidar):
    """
        Checks for a given vehicle in ego coordinate system, how many LiDAR hit there are in its bounding box.
        :param vehicle_pos: Relative position of the vehicle w.r.t. the ego
        :param vehicle_yaw: Relative orientation of the vehicle w.r.t. the ego
        :param extent: List, Extent of the bounding box
        :param lidar: LiDAR point cloud
        :return: Returns the number of LiDAR hits within the bounding box of the
        vehicle
        """

    rotation_matrix = np.array([[np.cos(vehicle_yaw), -np.sin(vehicle_yaw), 0.0],
                                [np.sin(vehicle_yaw), np.cos(vehicle_yaw), 0.0], [0.0, 0.0, 1.0]])

    # LiDAR in the with the vehicle as origin
    vehicle_lidar = (rotation_matrix.T @ (lidar - vehicle_pos).T).T

    # check points in bbox
    x, y, z = extent[0], extent[1], extent[2]
    num_points = ((vehicle_lidar[:, 0] < x) & (vehicle_lidar[:, 0] > -x) & (vehicle_lidar[:, 1] < y) &
                  (vehicle_lidar[:, 1] > -y) & (vehicle_lidar[:, 2] < z) & (vehicle_lidar[:, 2] > -z)).sum()
    return num_points

  def visualuize(self, rendered, visu_img):
    rendered = cv2.resize(rendered, dsize=(visu_img.shape[1], visu_img.shape[1]), interpolation=cv2.INTER_LINEAR)
    visu_img = cv2.cvtColor(visu_img, cv2.COLOR_BGR2RGB)

    final = np.concatenate((visu_img, rendered), axis=0)

    Image.fromarray(final).save(self.save_path / (f'{self.step:04}.jpg'))

  def _vehicle_obstacle_detected(self,
                                 vehicle_list=None,
                                 max_distance=None,
                                 up_angle_th=90,
                                 low_angle_th=0,
                                 lane_offset=0):
    """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
    self._use_bbs_detection = False
    self._offset = 0

    def get_route_polygon():
      route_bb = []
      extent_y = self._vehicle.bounding_box.extent.y
      r_ext = extent_y + self._offset
      l_ext = -extent_y + self._offset
      r_vec = ego_transform.get_right_vector()
      p1 = ego_location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
      p2 = ego_location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
      route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

      for wp, _ in self._local_planner.get_plan():
        if ego_location.distance(wp.transform.location) > max_distance:
          break

        r_vec = wp.transform.get_right_vector()
        p1 = wp.transform.location + carla.Location(r_ext * r_vec.x, r_ext * r_vec.y)
        p2 = wp.transform.location + carla.Location(l_ext * r_vec.x, l_ext * r_vec.y)
        route_bb.extend([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z]])

      # Two points don't create a polygon, nothing to check
      if len(route_bb) < 3:
        return None

      return Polygon(route_bb)

    if not vehicle_list:
      vehicle_list = self._world.get_actors().filter("*vehicle*")

    ego_transform = self._vehicle.get_transform()
    ego_location = ego_transform.location
    ego_wpt = self.world_map.get_waypoint(ego_location, lane_type=carla.libcarla.LaneType.Any)

    # Get the right offset
    if ego_wpt.lane_id < 0 and lane_offset != 0:
      lane_offset *= -1

    # Get the transform of the front of the ego
    ego_front_transform = ego_transform
    ego_front_transform.location += carla.Location(self._vehicle.bounding_box.extent.x *
                                                   ego_transform.get_forward_vector())

    opposite_invasion = abs(self._offset) + self._vehicle.bounding_box.extent.y > ego_wpt.lane_width / 2
    use_bbs = self._use_bbs_detection or opposite_invasion or ego_wpt.is_junction

    # Get the route bounding box
    route_polygon = get_route_polygon()

    for target_vehicle in vehicle_list:
      if target_vehicle.id == self._vehicle.id:
        continue

      target_transform = target_vehicle.get_transform()
      if target_transform.location.distance(ego_location) > max_distance:
        continue

      target_wpt = self.world_map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

      # General approach for junctions and vehicles invading other lanes due to the offset
      if (use_bbs or target_wpt.is_junction) and route_polygon:

        target_bb = target_vehicle.bounding_box
        target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
        target_list = [[v.x, v.y, v.z] for v in target_vertices]
        target_polygon = Polygon(target_list)

        if route_polygon.intersects(target_polygon):
          return (True, target_vehicle.id, compute_distance(target_vehicle.get_location(), ego_location))

      # Simplified approach, using only the plan waypoints (similar to TM)
      else:

        if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id + lane_offset:
          next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
          if not next_wpt:
            continue
          if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id + lane_offset:
            continue

        target_forward_vector = target_transform.get_forward_vector()
        target_extent = target_vehicle.bounding_box.extent.x
        target_rear_transform = target_transform
        target_rear_transform.location -= carla.Location(
            x=target_extent * target_forward_vector.x,
            y=target_extent * target_forward_vector.y,
        )

        if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
          return (True, target_vehicle.id, compute_distance(target_transform.location, ego_transform.location))

    return (False, None, -1)

  def _get_forward_speed(self, transform=None, velocity=None):
    """
        Calculate the forward speed of the vehicle based on its transform and velocity.

        Args:
            transform (carla.Transform, optional): The transform of the vehicle. If not provided, it will be obtained from the vehicle.
            velocity (carla.Vector3D, optional): The velocity of the vehicle. If not provided, it will be obtained from the vehicle.

        Returns:
            float: The forward speed of the vehicle in m/s.
        """
    if not velocity:
      velocity = self._vehicle.get_velocity()

    if not transform:
      transform = self._vehicle.get_transform()

    # Convert the velocity vector to a NumPy array
    velocity_np = np.array([velocity.x, velocity.y, velocity.z])

    # Convert rotation angles from degrees to radians
    pitch_rad = np.deg2rad(transform.rotation.pitch)
    yaw_rad = np.deg2rad(transform.rotation.yaw)

    # Calculate the orientation vector based on pitch and yaw angles
    orientation_vector = np.array(
        [np.cos(pitch_rad) * np.cos(yaw_rad),
         np.cos(pitch_rad) * np.sin(yaw_rad),
         np.sin(pitch_rad)])

    # Calculate the forward speed by taking the dot product of velocity and orientation vectors
    forward_speed = np.dot(velocity_np, orientation_vector)

    return forward_speed
  
  def add_disturbance(self, control, timestamp, input_data):
    steer_raw = control.steer
    steer_dist = self.disturbance_generater.run_step()
    steer_out = steer_raw + steer_dist

    control.steer = steer_out

    if self.conf_plot_control:
      if self.step_count % 5 == 0:
        if len(self.steer_raw_10s) >= 40:
          self.timestamp_10s.pop(0)
          self.steer_raw_10s.pop(0)
          self.steer_dist_10s.pop(0)
          self.steer_out_10s.pop(0)
          self.Throttle_10s.pop(0)
          self.brake_10s.pop(0)
          self.speed_10s.pop(0)


          self.timestamp_10s.append(timestamp)
          self.steer_raw_10s.append(steer_raw)
          self.steer_dist_10s.append(steer_dist)
          self.steer_out_10s.append(steer_out)
          self.Throttle_10s.append(control.throttle)
          self.brake_10s.append(control.brake)
          self.speed_10s.append(input_data['speed'][1]['speed'])
        else:
          self.timestamp_10s.append(timestamp)
          self.steer_raw_10s.append(steer_raw)
          self.steer_dist_10s.append(steer_dist)
          self.steer_out_10s.append(steer_out)
          self.Throttle_10s.append(control.throttle)
          self.brake_10s.append(control.brake)
          self.speed_10s.append(input_data['speed'][1]['speed'])
    return control
  
  def plot_status(self, input_data):

    if not hasattr(self, 'ax_plot'):
      self.fig_plot, self.ax_plot = plt.subplots(3, 1)

    self.ax_plot[0].clear()
    self.ax_plot[0].plot(self.timestamp_10s, self.steer_raw_10s, label='steer_raw')
    self.ax_plot[0].plot(self.timestamp_10s, self.steer_dist_10s, label='steer_dist')
    self.ax_plot[0].plot(self.timestamp_10s, self.steer_out_10s, label='steer_out')
    self.ax_plot[0].set_xlabel('Timestamp')
    self.ax_plot[0].set_ylabel('Steering')
    self.ax_plot[0].legend()
    self.ax_plot[0].grid(True)

    self.ax_plot[1].clear()
    self.ax_plot[1].plot(self.timestamp_10s, self.Throttle_10s, label='Throttle')
    self.ax_plot[1].plot(self.timestamp_10s, self.brake_10s, label='Brake')
    self.ax_plot[1].set_xlabel('Timestamp')
    self.ax_plot[1].set_ylabel('pct')
    self.ax_plot[1].legend()
    self.ax_plot[1].grid(True)

    self.ax_plot[2].clear()
    self.ax_plot[2].plot(self.timestamp_10s, [speed*3600/1000 for speed in self.speed_10s], label='Speed')
    self.ax_plot[2].set_xlabel('Timestamp')
    self.ax_plot[2].set_ylabel('kmph')
    self.ax_plot[2].legend()
    self.ax_plot[2].grid(True)

    # if not hasattr(self, 'ax_rgb'):
    #   self.fig_rgb, self.ax_rgb = plt.subplots(1, 1)
    # rgb = input_data['rgb'][1][:, :, :3]

    # _, rgb = cv2.imencode('.jpg', rgb)
    # rgb = cv2.imdecode(rgb, cv2.IMREAD_UNCHANGED)
    # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    # plt.imshow(rgb)

    plt.show(block=False)
    plt.pause(0.001)

    if self.commands[-2] == -1:
      tmp_command = "VOID"
    elif self.commands[-2] == 1:
      tmp_command = "LEFT"
    elif self.commands[-2] == 2:
      tmp_command = "RIGHT"
    elif self.commands[-2] == 3:
      tmp_command = "STRAIGHT"
    elif self.commands[-2] == 4:
      tmp_command = "LANEFOLLOW"
    elif self.commands[-2] == 5:
      tmp_command = "CHANGELANELEFT"
    elif self.commands[-2] == 6:
      tmp_command = "CHANGELANERIGHT"
    
    status_img = np.zeros((40, 200, 3), dtype=np.uint8)
    cv2.putText(status_img, tmp_command, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Command", status_img)
    cv2.waitKey(1)

class Disturbance_generater:
  class _disturbance_type(Enum):
    STEP = 0
    PULSE = 1
    SINE = 2

  def __init__(self, rate_per_sec, max_duration, max_magnitude, step_size):
    self.probability = rate_per_sec
    self.max_duration = max_duration
    self.max_magnitude = max_magnitude
    self.step_size = step_size

    self.dist_enabled = False
    self.disturbance_type = Disturbance_generater._disturbance_type.STEP
    self.disturbance_sign = 1
    self.magnitude = 0
    self.time = 0

  def run_step(self):
    if self.dist_enabled:
      if self.time >= self.duration:
        self.dist_enabled = False
        self.time = 0
        disturbance = 0

      else:
        if self.disturbance_type == Disturbance_generater._disturbance_type.STEP:
          disturbance = self.disturbance_sign * self.magnitude
          
        elif self.disturbance_type == Disturbance_generater._disturbance_type.PULSE:
          half_duration = self.duration/2
          if self.time <= half_duration:
            disturbance = self.disturbance_sign * self.time/half_duration * self.magnitude
          else:
            disturbance = self.disturbance_sign * (self.duration - self.time)/half_duration * self.magnitude
        
        elif self.disturbance_type == Disturbance_generater._disturbance_type.SINE:
          disturbance = self.disturbance_sign * self.magnitude * np.sin(self.time/self.duration * np.pi)

        self.time += self.step_size


    else:
      if random.random() < self.probability * self.step_size:
        self.dist_enabled = True
        self.time = 0
        self.disturbance_type = random.choice(list(Disturbance_generater._disturbance_type))
        self.disturbance_sign = random.choice([1, -1])
        self.magnitude = self.max_magnitude * random.random()
        self.duration = self.max_duration * random.random()
        disturbance = 0
      else:
        disturbance = 0
    
    return disturbance