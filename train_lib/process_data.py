import cv2
import gzip
import ujson
import numpy as np
from imgaug import augmenters as ia
from common.utils import post_augument, nomalize_image, bev_post_augument, rot_cam_coordi_sys, command_to_one_hot, circle_line_segment_intersection

class Process_Images:
  def __init__(self, 
              config, 
              shared_dict, 
              images, 
              images_augmented, 
              cameras, 
              validation=False):
    self.images = images
    self.images_augmented = images_augmented
    self.data_cache = shared_dict
    self.config = config
    self.validation = validation
    self.cameras = cameras

    self.image_augmenter_func = image_augmenter(config.color_aug_prob, cutout=config.use_cutout)

  def process_data(self, index, use_augment_sample, bev_postaug_rot):
    data = {}
    
    loaded_images, loaded_images_augmented = self.load_data(index)

    loaded_images = loaded_images[self.config.seq_len - 1]
    loaded_images_augmented = loaded_images_augmented [self.config.seq_len - 1]

    imgs = []
    rots = []
    trans = []
    intrins = []
    post_rots = []
    post_trans = []
    for idx, img in enumerate(loaded_images):
      if self.config.augment and use_augment_sample and idx == 0: #only for front cam
        img = loaded_images_augmented

      if self.config.use_color_aug and not self.validation:
        img = self.image_augmenter_func(image=img)

      if self.config.use_post_augment and not self.validation:
        img, post_rot, post_tran = post_augument(img, self.config.data_aug_conf, augmentation=True)
      else:
        img, post_rot, post_tran = post_augument(img, self.config.data_aug_conf, augmentation=False)

      img = nomalize_image(img, self.config.rgb_mean, self.config.rgb_std)

      

      camera = self.cameras[idx]
      imgs.append(np.transpose(img, (2, 0, 1)))
      intrins.append(camera.intrins)
      rots.append(camera.get_rot_mat())
      trans.append(camera.trans)
      post_rots.append(post_rot)
      post_trans.append(post_tran)

    rots = np.array(rots)
    trans = np.array(trans)
    if bev_postaug_rot is not None:
      trans, rots = rot_cam_coordi_sys(trans, rots, bev_postaug_rot)

    data["rgb_multi_cam"] = np.array(imgs)
    data['rgb'] = np.array(imgs[0])
    data["intrins"] = np.array(intrins)
    data["rots"] = rots
    data["trans"] = trans
    data["post_rots"] = np.array(post_rots)
    data["post_trans"] = np.array(post_trans)

    # plt.imshow(processed_image)
    # plt.show(block=False)

    return data
  
  def load_data(self, index):
    images = self.images[index]
    images_augmented = self.images_augmented[index]

    loaded_images = []
    loaded_images_augmented = []

    # Because the strings are stored as numpy byte objects we need to
    # convert them back to utf-8 strings

    for i in range(self.config.seq_len):
      imgs_each_camera = images[i]

      imgs = []
      for cmr_idx in range(len(self.cameras)):
        cache_key = str(imgs_each_camera[cmr_idx], encoding='utf-8')

        # Retrieve data from the disc cache
        if not self.data_cache is None and cache_key in self.data_cache:
          img = self.data_cache[cache_key]
          img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
      
        # Load data from the disc
        else:
          img_path = str(imgs_each_camera[cmr_idx], encoding='utf-8')
          camera_name = self.cameras[cmr_idx].name
          camera_dir = "/" + f"{camera_name:0>20}"
          img_path = img_path.replace(camera_dir, "/" + camera_name)#Decode padding for camera name
          img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # Store data inside disc cache
        if not self.data_cache is None:
          # We want to cache the images in jpg format instead of uncompressed, to reduce memory usage
          _, compressed_img = cv2.imencode('.jpg', img)
          self.data_cache[cache_key] = compressed_img
    
        imgs.append(img)

      img_augmented = None
      if self.config.augment:
        cache_key_aug = str(images_augmented[i], encoding='utf-8')

        # Retrieve data from the disc cache
        if not self.data_cache is None and cache_key_aug in self.data_cache:
          img_augmented = self.data_cache[cache_key_aug]
          img_augmented = cv2.imdecode(img_augmented, cv2.IMREAD_UNCHANGED)
        
        # Load data from the disc
        else:
          img_augmented = None
          img_augmented = cv2.imread(str(images_augmented[i], encoding='utf-8'), cv2.IMREAD_COLOR)

        # Store data inside disc cache
        if not self.data_cache is None:
          # We want to cache the images in jpg format instead of uncompressed, to reduce memory usage
          compressed_image_augmented = None
          _, compressed_image_augmented = cv2.imencode('.jpg', img_augmented)
          self.data_cache[cache_key_aug] = (compressed_image_augmented)

      loaded_images.append(imgs)
      loaded_images_augmented.append(img_augmented)
    return loaded_images, loaded_images_augmented
  
  def down_sampling(self, image, H, W):
    resized_image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
    return resized_image
  
class Process_Bev:
  def __init__(self, config, shared_dict, bev_semantics, bev_semantics_augmented, bev_config, validation=False):
    self.bev_semantics = bev_semantics
    self.bev_semantics_augmented = bev_semantics_augmented
    self.data_cache = shared_dict
    self.config = config
    self.bev_config = bev_config
    self.validation = validation

    self.bev_converter = np.uint8(bev_config["bev_converter"])

  def process_data(self, index, use_augment_sample):
    data = {}

    loaded_bev_semantics, loaded_bev_semantics_augmented = self.load_data(index)

    if self.config.augment and use_augment_sample:
      bev_semantics_i = self.bev_converter[loaded_bev_semantics_augmented[self.config.seq_len - 1]]  # pylint: disable=locally-disabled, unsubscriptable-object
    else:
      bev_semantics_i = self.bev_converter[loaded_bev_semantics[self.config.seq_len - 1]]  # pylint: disable=locally-disabled, unsubscriptable-object
    
    if self.config.use_bev_post_augment and not self.validation:
      bev_semantics_i, bev_postaug_rot = bev_post_augument(bev_semantics_i, self.config.data_aug_conf)
    else:
      bev_postaug_rot = None

    # NOTE the BEV label can unfortunately only be saved up to 2.0 ppm resolution. We upscale it here.
    # If you change these values you might need to change the up-scaling as well.
    assert self.bev_config["pixels_per_meter"] == 4.0
    assert self.bev_config["pixels_per_meter_collection"] == 2.0
    assert self.bev_config["lidar_resolution_width"] == 256
    assert self.bev_config["lidar_resolution_height"] == 256
    assert self.bev_config["max_x"] == 32
    assert self.bev_config["min_x"] == -32
    if self.bev_config["pixels_per_meter"] == 4.0:
      bev_semantics_i = bev_semantics_i[64:192, 64:192].repeat(2, axis=0).repeat(2, axis=1)

    for cls in self.config.ignore_class:
      bev_semantics_i[bev_semantics_i==cls] = 0
    
    # The indexing is an elegant way to down-sample the semantic images without interpolation or changing the dtype
    data['bev_semantic'] = bev_semantics_i

    return data, bev_postaug_rot
  
  def load_data(self, index):
    bev_semantics = self.bev_semantics[index]
    bev_semantics_augmented = self.bev_semantics_augmented[index]

    # load measurements
    loaded_bev_semantics = []
    loaded_bev_semantics_augmented = []

    # Because the strings are stored as numpy byte objects we need to
    # convert them back to utf-8 strings

    for i in range(self.config.seq_len):
      cache_key = str(bev_semantics[i], encoding='utf-8')

      # Retrieve data from the disc cache
      if not self.data_cache is None and cache_key in self.data_cache:
        bev_semantics_i, bev_semantics_augmented_i = self.data_cache[cache_key]
        bev_semantics_i = cv2.imdecode(bev_semantics_i, cv2.IMREAD_UNCHANGED)
        if self.config.augment:
          bev_semantics_augmented_i = cv2.imdecode(bev_semantics_augmented_i, cv2.IMREAD_UNCHANGED)

      # Load data from the disc
      else:
        bev_semantics_i = None
        bev_semantics_augmented_i = None

        bev_semantics_i = cv2.imread(str(bev_semantics[i], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
        if self.config.augment:
          bev_semantics_augmented_i = cv2.imread(str(bev_semantics_augmented[i], encoding='utf-8'),cv2.IMREAD_UNCHANGED)

      # Store data inside disc cache
      if not self.data_cache is None:
        # We want to cache the images in jpg format instead of uncompressed, to reduce memory usage
        compressed_bev_semantic_i = None
        compressed_bev_semantic_augmented_i = None

        _, compressed_bev_semantic_i = cv2.imencode('.png', bev_semantics_i)
        if self.config.augment:
          _, compressed_bev_semantic_augmented_i = cv2.imencode('.png', bev_semantics_augmented_i)

        self.data_cache[cache_key] = (compressed_bev_semantic_i, compressed_bev_semantic_augmented_i)

      loaded_bev_semantics.append(bev_semantics_i)
      loaded_bev_semantics_augmented.append(bev_semantics_augmented_i)

    return loaded_bev_semantics, loaded_bev_semantics_augmented
  
class Process_Measurements:
  def __init__(self, 
              config, 
              shared_dict, 
              measurements,
              sample_start,
              validation=False):
    self.data_cache = shared_dict
    self.config = config
    self.validation = validation
    self.measurements = measurements
    self.sample_start = sample_start

    self.target_speed_bins = np.array(config.target_speed_bins)
    self.angle_bins = np.array(config.angle_bins)

  def load_data(self, index):
    """Load measurement JSON files from disk with caching."""
    measurements = self.measurements[index]
    sample_start = self.sample_start[index]

    loaded_measurements = []

    # Load current (and past) frames
    for i in range(self.config.seq_len):
      measurement_file = str(measurements[0], encoding='utf-8') + (f'/{(sample_start + i):04}.json.gz')
      if (self.data_cache is not None) and (measurement_file in self.data_cache):
        measurements_i = self.data_cache[measurement_file]
      else:
        with gzip.open(measurement_file, 'rt', encoding='utf-8') as f:
          measurements_i = ujson.load(f)

        if self.data_cache is not None:
          self.data_cache[measurement_file] = measurements_i

      loaded_measurements.append(measurements_i)

    # Load future frames for waypoint prediction
    if self.config.use_wp_gru:
      end = self.config.pred_len + self.config.seq_len
      start = self.config.seq_len
    else:
      end = 0
      start = 0
    for i in range(start, end, self.config.wp_dilation):
      measurement_file = str(measurements[0], encoding='utf-8') + (f'/{(sample_start + i):04}.json.gz')
      if (self.data_cache is not None) and (measurement_file in self.data_cache):
        measurements_i = self.data_cache[measurement_file]
      else:
        with gzip.open(measurement_file, 'rt', encoding='utf-8') as f:
          measurements_i = ujson.load(f)

        if self.data_cache is not None:
          self.data_cache[measurement_file] = measurements_i

      loaded_measurements.append(measurements_i)

    return loaded_measurements

  def process_data(self, index, use_augment_sample):
    """Process measurement data and return a dict of processed values."""
    loaded_measurements = self.load_data(index)
    current_measurement = loaded_measurements[self.config.seq_len - 1]

    # Determine augmentation parameters
    if use_augment_sample:
      aug_rotation = current_measurement['augmentation_rotation']
      aug_translation = current_measurement['augmentation_translation']
    else:
      aug_rotation = 0.0
      aug_translation = 0.0

    # Waypoints (if use_wp_gru)
    ego_waypoints = None
    if self.config.use_wp_gru:
      waypoints = self._get_waypoints(loaded_measurements[self.config.seq_len - 1:],
                                      y_augmentation=aug_translation,
                                      yaw_augmentation=aug_rotation)
      ego_waypoints = np.array(waypoints)

    # Scalar values
    brake = current_measurement['brake']
    target_speed_index, angle_index = self._get_indices_speed_angle(
        target_speed=current_measurement['target_speed'],
        brake=brake,
        angle=current_measurement['angle'])
    target_speed_twohot = self._get_two_hot_encoding(
        current_measurement['target_speed'],
        self.config.target_speeds,
        brake)

    # Command one-hot
    command = command_to_one_hot(current_measurement['command'])
    next_command = command_to_one_hot(current_measurement['next_command'])

    # Route processing
    route = current_measurement['route']
    if len(route) < self.config.num_route_points:
      num_missing = self.config.num_route_points - len(route)
      route = np.array(route)
      route = np.vstack((route, np.tile(route[-1], (num_missing, 1))))
    else:
      route = np.array(route[:self.config.num_route_points])

    route = self._augment_route(route, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
    if self.config.smooth_route:
      route = self._smooth_path(route)

    # Target point processing
    target_point = np.array(current_measurement['target_point'])
    target_point = self._augment_target_point(target_point,
                                              y_augmentation=aug_translation,
                                              yaw_augmentation=aug_rotation)

    target_point_next = np.array(current_measurement['target_point_next'])
    target_point_next = self._augment_target_point(target_point_next,
                                                   y_augmentation=aug_translation,
                                                   yaw_augmentation=aug_rotation)

    # --- Populate data dict ---
    data = {
        'brake': brake,
        'angle_index': angle_index,
        'target_speed': target_speed_index,
        'target_speed_twohot': target_speed_twohot,
        'steer': current_measurement['steer'],
        'throttle': current_measurement['throttle'],
        'light': current_measurement['light_hazard'],
        'stop_sign': current_measurement['stop_sign_hazard'],
        'junction': current_measurement['junction'],
        'speed': current_measurement['speed'],
        'theta': current_measurement['theta'],
        'command': command,
        'next_command': next_command,
        'route': route,
        'target_point': target_point,
        'target_point_next': target_point_next,
    }
    if ego_waypoints is not None:
      data['ego_waypoints'] = ego_waypoints

    return data

  # ---------------------------------------------------------------------------
  # Internal utility methods
  # ---------------------------------------------------------------------------

  @staticmethod
  def _augment_route(route, y_augmentation=0.0, yaw_augmentation=0.0):
    """Apply rotation + translation augmentation to route points."""
    aug_yaw_rad = np.deg2rad(yaw_augmentation)
    rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)],
                                [np.sin(aug_yaw_rad),  np.cos(aug_yaw_rad)]])
    translation = np.array([[0.0, y_augmentation]])
    route_aug = (rotation_matrix.T @ (route - translation).T).T
    return route_aug

  @staticmethod
  def _augment_target_point(target_point, y_augmentation=0.0, yaw_augmentation=0.0):
    """Apply rotation + translation augmentation to a single 2D point."""
    aug_yaw_rad = np.deg2rad(yaw_augmentation)
    rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)],
                                [np.sin(aug_yaw_rad),  np.cos(aug_yaw_rad)]])
    translation = np.array([[0.0], [y_augmentation]])
    pos = np.expand_dims(target_point, axis=1)
    target_point_aug = rotation_matrix.T @ (pos - translation)
    return np.squeeze(target_point_aug)

  def _get_waypoints(self, measurements, y_augmentation=0.0, yaw_augmentation=0.0):
    """Transform waypoints to be origin at ego_matrix."""
    origin = measurements[0]
    origin_matrix = np.array(origin['ego_matrix'])[:3]
    origin_translation = origin_matrix[:, 3:4]
    origin_rotation = origin_matrix[:, :3]

    waypoints = []
    for index in range(self.config.seq_len, len(measurements)):
      waypoint = np.array(measurements[index]['ego_matrix'])[:3, 3:4]
      waypoint_ego_frame = origin_rotation.T @ (waypoint - origin_translation)
      waypoints.append(waypoint_ego_frame[:2, 0])

    # Data augmentation
    waypoints_aug = []
    aug_yaw_rad = np.deg2rad(yaw_augmentation)
    rotation_matrix = np.array([[np.cos(aug_yaw_rad), -np.sin(aug_yaw_rad)],
                                [np.sin(aug_yaw_rad),  np.cos(aug_yaw_rad)]])
    translation = np.array([[0.0], [y_augmentation]])
    for waypoint in waypoints:
      pos = np.expand_dims(waypoint, axis=1)
      waypoint_aug = rotation_matrix.T @ (pos - translation)
      waypoints_aug.append(np.squeeze(waypoint_aug))

    return waypoints_aug

  def _get_indices_speed_angle(self, target_speed, brake, angle):
    """Convert target_speed and angle to bin indices."""
    target_speed_index = np.digitize(x=target_speed, bins=self.target_speed_bins)
    if brake:
      target_speed_index = 0
    else:
      target_speed_index += 1

    angle_index = np.digitize(x=angle, bins=self.angle_bins)
    return target_speed_index, angle_index

  @staticmethod
  def _get_two_hot_encoding(target_speed, config_target_speeds, brake):
    """Compute two-hot soft label encoding for target speed."""
    if target_speed < 0:
      raise ValueError('Target speed value must be non-negative for two-hot encoding.')
    label = np.zeros((len(config_target_speeds),))
    if brake:
      label[0] = 1.0
    else:
      if not np.any(np.array(config_target_speeds) > target_speed):
        label[-1] = 1.0
      else:
        upper_ind = np.argmax(np.array(config_target_speeds) > target_speed)
        lower_ind = upper_ind - 1
        lower_val = config_target_speeds[lower_ind]
        upper_val = config_target_speeds[upper_ind]
        lower_weight = (upper_val - target_speed) / (upper_val - lower_val)
        upper_weight = (target_speed - lower_val) / (upper_val - lower_val)
        label[lower_ind] = lower_weight
        label[upper_ind] = upper_weight
    return label

  def _smooth_path(self, route):
    """Remove duplicates and re-interpolate route at uniform spacing."""
    _, indices = np.unique(route, return_index=True, axis=0)
    route = route[np.sort(indices)]
    interpolated_route_points = self._iterative_line_interpolation(route)
    return interpolated_route_points

  def _iterative_line_interpolation(self, route):
    """Interpolate route points at 1.0m intervals using circle-line intersection."""
    interpolated_route_points = []

    min_distance = self.config.dense_route_planner_min_distance
    target_first_distance = 2.5
    last_interpolated_point = np.array([0.0, 0.0])
    current_route_index = 0
    current_point = route[current_route_index]
    last_point = np.array([0.0, 0.0])
    first_iteration = True

    while len(interpolated_route_points) < self.config.num_route_points:
      if not first_iteration:
        current_route_index += 1
        last_point = current_point

      if current_route_index < route.shape[0]:
        current_point = route[current_route_index]
        intersection = circle_line_segment_intersection(
            circle_center=last_interpolated_point,
            circle_radius=min_distance if not first_iteration else target_first_distance,
            pt1=last_interpolated_point,
            pt2=current_point,
            full_line=True)
      else:
        current_point = route[-1]
        last_point = route[-2]
        intersection = circle_line_segment_intersection(
            circle_center=last_interpolated_point,
            circle_radius=min_distance,
            pt1=last_point,
            pt2=current_point,
            full_line=True)

      if len(intersection) > 1:
        point_1 = np.array(intersection[0])
        point_2 = np.array(intersection[1])
        direction = current_point - last_point
        dot_p1_to_last = np.dot(point_1, direction)
        dot_p2_to_last = np.dot(point_2, direction)

        if dot_p1_to_last > dot_p2_to_last:
          intersection_point = point_1
        else:
          intersection_point = point_2
        add_point = True
      elif len(intersection) == 1:
        intersection_point = np.array(intersection[0])
        add_point = True
      else:
        add_point = False
        raise Exception("No intersection found. This should never occur.")

      if add_point:
        last_interpolated_point = intersection_point
        interpolated_route_points.append(intersection_point)
        min_distance = 1.0

      first_iteration = False

    interpolated_route_points = np.array(interpolated_route_points)
    return interpolated_route_points
  
def image_augmenter(prob=0.2, cutout=False):
  augmentations = [
      ia.Sometimes(prob, ia.GaussianBlur((0, 1.0))),
      ia.Sometimes(prob, ia.AdditiveGaussianNoise(loc=0, scale=(0., 0.05 * 255), per_channel=0.5)),
      ia.Sometimes(prob, ia.Dropout((0.01, 0.1), per_channel=0.5)),  # Strong
      ia.Sometimes(prob, ia.Multiply((1 / 1.2, 1.2), per_channel=0.5)),
      ia.Sometimes(prob, ia.LinearContrast((1 / 1.2, 1.2), per_channel=0.5)),
      ia.Sometimes(prob, ia.Grayscale((0.0, 0.5))),
      ia.Sometimes(prob, ia.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)),
  ]

  if cutout:
    augmentations.append(ia.Sometimes(prob, ia.arithmetic.Cutout(squared=False)))

  augmenter = ia.Sequential(augmentations, random_order=True)

  return augmenter