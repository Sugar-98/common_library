from common.config.DataLoader_conf import DataLoader_conf
from common.config.DataAgent_conf import DataAgent_conf
import os

class Train_conf:
  def __init__(self):
    # -----------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------
    self.local_rank = -999
    self.id = 'transfuser'  # Unique experiment identifier.
    self.epochs = 31  # Number of epochs to train
    self.lr = 3e-4  # Learning rate used for training
    self.batch_size = 16  # Batch size used during training
    self.logdir = ''  # Directory to log data to.
    self.load_file = None  # File to continue training from
    self.setting = 'all'  # Setting used for training
    self.root_dir = ''  # Dataset root dir
    # When to reduce the learning rate for the first and second  time
    self.schedule_reduce_epoch_01 = 30
    self.schedule_reduce_epoch_02 = 40
    self.val_every = 5  # Validation frequency in epochs
    self.sync_batch_norm = 0  # Whether batch norm was synchronized between GPUs
    # Whether zero_redundancy_optimizer was used during training
    self.zero_redundancy_optimizer = 1
    self.use_disk_cache = 0  # Whether disc cache was used during training
    self.detect_boxes = 0  # Whether to use the bounding box auxiliary task
    # Number of route points we use for prediction in TF or input in planT
    self.num_route_points = 20
    self.learn_origin = 1  # Whether to learn the origin of the waypoints or use 0 / 0
    # At which interval to save debug files to disk during training
    self.train_debug_save_freq = 1
    self.backbone = 'transFuser'  # Vision backbone architecture used
    self.use_velocity = 0  # Whether to use the velocity as input to the network
    self.image_architecture = 'regnety_032'  # Image architecture used in the backbone resnet34, regnety_032
    self.lidar_architecture = 'regnety_032'  # LiDAR architecture used in the backbone resnet34, regnety_032
    # Whether to classify target speeds and regress a path as output representation.
    self.use_controller_input_prediction = False
    # Whether to use the direct control predictions for driving
    self.inference_direct_controller = True
    # Label smoothing applied to the cross entropy losses
    self.label_smoothing_alpha = 0.1
    # Whether to use focal loss instead of cross entropy for classification
    self.use_focal_loss = False
    # Gamma hyperparameter of focal loss
    self.focal_loss_gamma = 2.0
    # Learning rate decay, applied when using multi-step scheduler
    self.multi_step_lr_decay = 0.1
    # Whether to use a cosine schedule instead of the linear one.
    self.use_cosine_schedule = False
    # Epoch of the first restart
    self.cosine_t0 = 1
    # Multiplier applied to t0 after every restart
    self.cosine_t_mult = 2
    # Weights applied to each of these losses, when combining them
    self.detailed_loss_weights = {
        'loss_wp': 1.0,
        'loss_target_speed': 1.0,
        'loss_checkpoint': 1.0,
        'loss_semantic': 1.0,
        'loss_bev_semantic': 1.0,
        'loss_depth': 1.0,
        'loss_center_heatmap': 1.0,
        'loss_wh': 1.0,
        'loss_offset': 1.0,
        'loss_yaw_class': 1.0,
        'loss_yaw_res': 1.0,
        'loss_velocity': 1.0,
        'loss_brake': 1.0,
        'loss_forcast': 0.2,
        'loss_selection': 0.0,
    }
    self.root_dir = ''
    # NOTE currently leads to inf gradients do not use! Whether to use automatic mixed precision during training.
    self.use_amp = 0
    self.use_grad_clip = 0  # Whether to clip the gradients
    self.grad_clip_max_norm = 1.0  # Max value for the gradients if gradient clipping is used.
    self.lidar_aug_prob = 1.0  # Probability with which data augmentation is applied to the LiDAR image.
    self.freeze_backbone = False  # Whether to freeze the image backbone during training. Useful for 2 stage training.
    self.learn_multi_task_weights = False  # Whether to learn the multi-task weights
    self.use_depth = False  # Whether to use depth prediction as auxiliary loss for training.
    self.continue_epoch = True  # Whether to continue the training from the loaded epoch or from 0.

    self.smooth_route = True  # Whether to smooth the route points with a spline.
    self.ignore_index = -999  # Index to ignore for future bounding box prediction task.
    self.use_speed_weights = False  # Whether to weight target speed classes
    self.use_optim_groups = False  # Whether to use optimizer groups to exclude some parameters from weight decay
    self.weight_decay = 0.01  # Weight decay coefficient used during training
    self.use_label_smoothing = False  # Whether to use label smoothing in the classification losses
    self.use_twohot_target_speeds = False  # Whether to use two hot encoding for the target speed classification
    self.compile = False  # Whether to apply torch.compile to the model.
    self.compile_mode = 'default'  # Compile mode for torch.compile
    self.plot_metrics = [
      'm_precision', 
      'm_recall', 
      'm_f1', 
      'mIoU'
    ]# Metrics which are plotted during each epochs
    self.val_towns = [13]

  def initialize(self, root_dir='', setting='all', **kwargs):
    self.DataAgent_config = DataAgent_conf()
    self.DataLoader_config = DataLoader_conf(self.DataAgent_config)

    for k, v in kwargs.items():
      setattr(self, k, v)

    self.root_dir = root_dir

    if setting == 'all':
      pass
    elif setting == '13_withheld':
      self.val_towns.append(13)
    elif setting == '12_only':
      self.val_towns.append(1)
      self.val_towns.append(2)
      self.val_towns.append(3)
      self.val_towns.append(4)
      self.val_towns.append(5)
      self.val_towns.append(6)
      self.val_towns.append(7)
      self.val_towns.append(10)
      self.val_towns.append(11)
      self.val_towns.append(13)
      self.val_towns.append(15)

    elif setting == 'eval':
      return
    else:
      raise ValueError(f'Error: Selected setting: {setting} does not exist.')

    print('Setting: ', setting)
    self.data_roots = []
    for td_path in self.root_dir:
      self.data_roots = self.data_roots + [os.path.join(td_path, name) for name in os.listdir(td_path)]
