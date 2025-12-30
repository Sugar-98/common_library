


class DataLoader_conf:
  def __init__(self, DataAgent_config):
    # -----------------------------------------------------------------------------
    # Dataloader
    # -----------------------------------------------------------------------------
    self.num_repetitions = 2  # How many repetitions of the dataset we train with.
    self.carla_fps = 20  # Simulator Frames per second
    self.seq_len = 1  # input timesteps
    # use different seq len for image and lidar
    self.img_seq_len = 1
    self.lidar_seq_len = 1
    # Number of initial frames to skip during data loading
    self.skip_first = int(2.5 * self.carla_fps) // DataAgent_config.data_save_freq
    self.pred_len = int(2.0 * self.carla_fps) // DataAgent_config.data_save_freq  # number of future waypoints predicted

    # Crop the BEV semantics, bounding boxes and LiDAR range to the values above. Also affects self.lidar_vert_anchors
    self.crop_bev = False
    # If true, cuts BEV off behind the vehicle. If False, cuts off front and back symmetrically
    self.crop_bev_height_only_from_behind = False
    # Number of LiDAR hits a bounding box needs for it to be a valid label
    self.num_lidar_hits_for_detection_walker = 1
    self.num_lidar_hits_for_detection_car = 1
    # How many pixels make up 1 meter in BEV grids
    # 1 / pixels_per_meter = size of pixel in meters
    self.pixels_per_meter = 4.0

    # Max number of LiDAR points per pixel in voxelized LiDAR
    self.hist_max_per_pixel = 5
    # Height at which the LiDAR points are split into the 2 channels.
    # Is relative to lidar_pos[2]
    self.lidar_split_height = 0.2
    self.realign_lidar = True
    self.use_ground_plane = False
    # Max and minimum LiDAR ranges used for voxelization
    self.min_x = -32
    self.max_x = 32
    if self.crop_bev and not self.crop_bev_height_only_from_behind:
      assert self.max_x == -self.min_x  # If we cut the bev semantics symetrically, we also need a symmetric lidar range
    self.min_y = -32
    self.max_y = 32
    self.min_z = -4
    self.max_z = 4
    self.min_z_projection = -10
    self.max_z_projection = 14

    # Angle bin thresholds
    self.angle_bins = [-0.375, -0.125, 0.125, 0.375]
    # Discrete steering angles
    self.angles = [-0.5, -0.25, 0.0, 0.25, 0.5]
    # Whether to estimate the class weights or use the default from the config.
    self.estimate_class_distributions = False
    self.estimate_semantic_distribution = False
    # Class weights applied to the cross entropy losses
    self.angle_weights = [
        204.25901201602136, 7.554315623148331, 0.21388916461734406, 5.476446162657503, 207.86684782608697
    ]
    # We don't use weighting here
    self.semantic_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    self.bev_semantic_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # v4 target speeds (0.72*speed limits) plus extra classes for obstacle scenarios and intersections
    self.target_speeds = [0.0, 4.0, 8.0, 10, 13.88888888, 16, 17.77777777, 20]

    self.target_speed_bins = [x + 0.001 for x in self.target_speeds[1:]]  # not used with two hot encodings
    self.target_speed_weights = [1.0] * (len(self.target_speeds))

    self.rgb_mean = [0.485, 0.456, 0.406]
    self.rgb_std  = [0.229, 0.224, 0.225]

    resize_lim=(0.193, 0.225)#リサイズ比率の範囲 Low, High
    final_dim=(128, 352)#加工後の画像サイズ
    bot_pct_lim=(0.0, 0.22)#下端の切る範囲 Low, High
    rot_lim=(-5.4, 5.4)#回転の範囲
    bev_rot_lim=(-10, 10)
    H=DataAgent_config.cameras[0].H
    W=DataAgent_config.cameras[0].W
    ncams=len(DataAgent_config.cameras)
    rand_flip=True#左右反転
    
    self.data_aug_conf = {
      'resize_lim': resize_lim,
      'final_dim': final_dim,
      'rot_lim': rot_lim,
      'bev_rot_lim' : bev_rot_lim,
      'H': H, 'W': W,
      'rand_flip': rand_flip,
      'bot_pct_lim': bot_pct_lim,
      'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
      'Ncams': ncams,
    }
    self.MaskScoreCheck_for_Dataset = True
    self.num_max_data_train = 1000
    self.num_max_data_val = 100
    self.train_sampling_rate = 1  # We train on every n th sample on the route
    self.forcast_time = 0.5  # Number of seconds we forcast into the future
    self.carla_fps = 20  # Simulator Frames per second
    self.min_abs_speed = None #Reduce low speed data
    self.min_abs_speed_rate = 0.90 #Reduce rate
    self.low_steer_threshold = None # Thewshold to detect straight run
    self.reduce_straight_run_rate = [1/1.3, 1/1.5, 1/1.6, 1/6, 1/3] #rate for ignore low steer data
    self.ignore_command = []
    self.use_bev_semantic = True  # Whether to use bev semantic segmentation as auxiliary loss for training.
    self.augment_percentage = 0  # Probability of the augmented sample being used.
    self.augment = 0  # Whether to use rotation and translation augmentation
    self.use_color_aug = 1  # Whether to apply image color based augmentations
    self.color_aug_prob = 0.5  # With which probability to apply the different image color augmentations.
    self.use_cutout = False  # Whether to use cutout as a data augmentation technique during training.
    self.use_post_augment = False
    self.use_bev_post_augment = False
    self.ignore_class = []
    self.use_wp_gru = False  # Whether to use the WP output GRU.