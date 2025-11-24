
from team_code.data_agent import DataAgent
from team_code.kinematic_bicycle_model import KinematicBicycleModel
from team_code.lateral_controller import LateralPIDController
from team_code.scenario_logger import ScenarioLogger
from scenario_runner.srunner.scenariomanager.carla_data_provider import CarlaDataProvider
import pathlib
from enum import Enum
import os
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import cv2
import carla
from leaderboard.leaderboard.autoagents import autonomous_agent

from common.config import GlobalConfig

def get_entry_point():
  return 'DataAgent_modified'

class DataAgent_modified(DataAgent):
  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    #-------configuration for data agent----------------------
    self.conf_enable_disturbance = False #Add steering disturbance for agent
    self.conf_plot_control = False #Plot control signals during runtime
    #---------------------------------------------------------

    super.setup(path_to_conf_file, route_index=None, traffic_manager=None)
    #--------overwrite config----------------
    self.config = GlobalConfig()
    self.ego_model = KinematicBicycleModel(self.config)
    self.vehicle_model = KinematicBicycleModel(self.config)
    self._turn_controller = LateralPIDController(self.config)
    #----------------------------------------

    if self.save_path is not None and self.datagen:
      for idx, camera_name in enumerate(self.config.camera_name):
        (self.save_path / 'rgb' / camera_name).mkdir()

    if self.conf_enable_disturbance:
      self.disturbance_generater = Disturbance_generater(rate_per_sec = 1, max_duration = 2, max_magnitude = 0.5, step_size = 1/20)

    if self.conf_plot_control:
      self.step_count = 0
      self.timestamp_10s = []
      self.steer_raw_10s = []
      self.steer_dist_10s = []
      self.steer_out_10s = []
      self.Throttle_10s = []
      self.brake_10s = []
      self.speed_10s = []

    

    #-----------over write parent parameters----------------------
    if self.conf_enable_disturbance:
      #Reduce responce to disturbance
      self._turn_controller.lateral_pid_kd = 0
    

    #extend sensor timeout because of machine resources
    self.sensor_interface._queue_timeout = 100

  def sensors(self):
    result = super.sensors(self)
    for idx, camera_name in enumerate(self.config.camera_name):
      tmp_camera = {
      'type': 'sensor.camera.rgb',
      'x': self.config.camera_pos[idx][0],
      'y': self.config.camera_pos[idx][1],
      'z': self.config.camera_pos[idx][2],
      'roll': self.config.camera_rot_0[idx][0],
      'pitch': self.config.camera_rot_0[idx][1],
      'yaw': self.config.camera_rot_0[idx][2],
      'width': self.config.camera_width,
      'height': self.config.camera_height,
      'fov': self.config.camera_fov,
      'id': camera_name
      }
      result.append(tmp_camera)
    return result
  
  def tick(self, input_data):
    super.tick(self, input_data)
    

  def run_step(self, input_data, timestamp, sensors=None, plant=False):
    control = super().run_step(input_data, timestamp, sensors=sensors, plant=plant)

    if self.conf_enable_disturbance:
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

      self.plot_status(input_data)

    self.step_count += 1

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
