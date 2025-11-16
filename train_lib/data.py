"""
Code that loads the dataset for training.
"""

import os
import ujson
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import cv2
import gzip
import team_code.transfuser_utils as t_u
import team_code.gaussian_target as g_t
import random
from sklearn.utils.class_weight import compute_class_weight
from imgaug import augmenters as ia
import pickle
import re
import matplotlib.pyplot as plt
from train_lib.process_data import Process_Images, Process_Bev


class CARLA_Data(Dataset):  # pylint: disable=locally-disabled, invalid-name
  """
    Custom dataset that dynamically loads a CARLA dataset from disk.
    """

  def __init__(self,
               root,
               config,
               estimate_class_distributions=False,
               estimate_sem_distribution=False,
               shared_dict=None,
               rank=0,
               validation=False):
    self.config = config
    self.validation = validation
    assert config.img_seq_len == 1

    self.data_cache = shared_dict
    self.target_speed_bins = np.array(config.target_speed_bins)
    self.angle_bins = np.array(config.angle_bins)
    self.converter = np.uint8(config.converter)

    self.images = []
    self.images_augmented = []
    self.semantics = []
    self.semantics_augmented = []
    self.bev_semantics = []
    self.bev_semantics_augmented = []
    self.depth = []
    self.depth_augmented = []
    self.lidars = []
    self.boxes = []
    self.future_boxes = []
    self.measurements = []
    self.sample_start = []

    self.temporal_lidars = []
    self.temporal_measurements = []
    self.num_data = 0

    # Initialize with 1 example per class
    self.angle_distribution = np.arange(len(config.angles)).tolist()
    self.speed_distribution = np.arange(len(config.target_speeds)).tolist()
    self.semantic_distribution = np.arange(len(config.semantic_weights)).tolist()
    total_routes = 0
    trainable_routes = 0
    skipped_routes = 0

    # loops over the scenarios given in root (which is a list of the scenario folders)
    for sub_root in tqdm(root, file=sys.stdout, disable=rank != 0):
      #folder_name = os.path.basename(sub_root)
      #if self.validation and folder_name != 'validation':
      #  continue
      #elif not self.validation and folder_name == 'validation':
      #  continue

      # list subdirectories in root
      routes = next(os.walk(sub_root))[1]

      for route in routes:  # loop over individual routes within this scenario folder
        
        repetition = int(re.search('_Rep(\\d+)', route).group(1))
        if repetition >= self.config.num_repetitions:
          continue

        town = int(re.search('Town(\\d+)', route).group(1))
        if self.config.val_towns:
          if self.validation and (town not in self.config.val_towns):
            continue
          elif not self.validation and (town in self.config.val_towns):
            continue

        route_dir = sub_root + '/' + route
        total_routes += 1

        if route.startswith('FAILED_') or not os.path.isfile(route_dir + '/results.json.gz'):
          skipped_routes += 1
          continue

        # We skip data where the expert did not achieve perfect driving score (except for min speed infractions)
        with gzip.open(route_dir + '/results.json.gz', 'rt', encoding='utf-8') as f:
          results_route = ujson.load(f)
        if self.config.MaskScoreCheck_for_Dataset:
          condition1 = False
        else:
          condition1 = (results_route['scores']['score_composed'] < 100.0 and \
          not (results_route['num_infractions'] == len(results_route['infractions']['min_speed_infractions'])))
        
        condition2 = results_route['status'] == 'Failed - Agent couldn\'t be set up'
        condition3 = results_route['status'] == 'Failed'
        condition4 = results_route['status'] == 'Failed - Simulation crashed'
        condition5 = results_route['status'] == 'Failed - Agent crashed'
        if condition1 or condition2 or condition3 or condition4 or condition5:
          continue

        trainable_routes += 1

        lidar_dir = route_dir + '/lidar'
        measurements_dir = route_dir + '/measurements'
        if os.path.exists(lidar_dir):
          num_seq = len(os.listdir(lidar_dir))
        elif os.path.exists(measurements_dir):
          num_seq = len(os.listdir(measurements_dir))
        else:
          skipped_routes += 1
          continue

        # If we are using checkpoints to predict the path, we can use all of the frames, otherwise we need to subtract
        # pred_len so that we have enough waypoint labels
        last_frame = num_seq - (self.config.seq_len - 1) - (0 if not self.config.use_wp_gru else self.config.pred_len)
        for seq in range(config.skip_first, last_frame):
          if self.validation:
            if not self.config.num_max_data_val is None and self.config.num_max_data_val <= self.num_data:
                    break
          else:
            if not self.config.num_max_data_train is None and self.config.num_max_data_train <= self.num_data:
                    break
          
          if seq % config.train_sampling_rate != 0:
            continue

          # load input seq and pred seq jointly
          image = []
          image_augmented = []
          semantic = []
          semantic_augmented = []
          bev_semantic = []
          bev_semantic_augmented = []
          depth = []
          depth_augmented = []
          lidar = []
          box = []
          future_box = []
          measurement = []

          # Loads the current (and past) frames (if seq_len > 1)
          for idx in range(self.config.seq_len):
            
            if not self.config.use_plant:

              rgbs = []
              for camera in self.config.cameras:
                camera_dir = "/" + f"{camera.name:0>20}"
                rgbs.append(route_dir + '/rgb' + camera_dir + (f'/{(seq + idx):04}.jpg'))
              image.append(rgbs)
              image_augmented.append(route_dir + '/rgb_augmented' + (f'/{(seq + idx):04}.jpg'))
              semantic.append(route_dir + '/semantics' + (f'/{(seq + idx):04}.png'))
              semantic_augmented.append(route_dir + '/semantics_augmented' + (f'/{(seq + idx):04}.png'))
              bev_semantic.append(route_dir + '/bev_semantics' + (f'/{(seq + idx):04}.png'))
              bev_semantic_augmented.append(route_dir + '/bev_semantics_augmented' + (f'/{(seq + idx):04}.png'))
              depth.append(route_dir + '/depth' + (f'/{(seq + idx):04}.png'))
              depth_augmented.append(route_dir + '/depth_augmented' + (f'/{(seq + idx):04}.png'))
              lidar.append(route_dir + '/lidar' + (f'/{(seq + idx):04}.laz'))

              if estimate_sem_distribution:
                semantics_i = self.converter[cv2.imread(semantic[-1], cv2.IMREAD_UNCHANGED)]  # pylint: disable=locally-disabled, unsubscriptable-object
                self.semantic_distribution.extend(semantics_i.flatten().tolist())

            forcast_step = int(config.forcast_time / (config.data_save_freq / config.carla_fps) + 0.5)

            box.append(route_dir + '/boxes' + (f'/{(seq + idx):04}.json.gz'))
            future_box.append(route_dir + '/boxes' + (f'/{(seq + idx + forcast_step):04}.json.gz'))

          # we only store the root and compute the file name when loading,
          # because storing 40 * long string per sample can go out of memory.
          measurement.append(route_dir + '/measurements')

          with gzip.open(measurement[-1] + f'/{(seq):04}.json.gz', 'rt', encoding='utf-8') as f:
              measurements_i = ujson.load(f)

          if self.config.min_abs_speed is not None:
            speed = measurements_i['speed']
            if (-1*self.config.min_abs_speed < speed) & (speed < self.config.min_abs_speed):
              if random.random() <= self.config.min_abs_speed_rate:
                continue
            
          if (self.config.low_steer_threshold is not None) & (self.config.reduce_straight_run_rate is not None):
            steer = measurements_i['steer']
            if (-1*self.config.low_steer_threshold[3] < steer) & (steer < self.config.low_steer_threshold[4]):
              if (-1*self.config.low_steer_threshold[3] < steer) & (steer < self.config.low_steer_threshold[3]):
                if (-1*self.config.low_steer_threshold[2] < steer) & (steer < self.config.low_steer_threshold[2]):
                  if (-1*self.config.low_steer_threshold[1] < steer) & (steer < self.config.low_steer_threshold[1]):
                    if (-1*self.config.low_steer_threshold[0] < steer) & (steer < self.config.low_steer_threshold[0]):
                      if random.random() <= self.config.reduce_straight_run_rate[0]:
                        continue
                    else:
                      if random.random() <= self.config.reduce_straight_run_rate[1]:
                        continue
                  else:
                    if random.random() <= self.config.reduce_straight_run_rate[2]:
                      continue
                else:
                  if random.random() <= self.config.reduce_straight_run_rate[3]:
                    continue
              if random.random() <= self.config.reduce_straight_run_rate[4]:
                continue

          if measurements_i['command'] in self.config.ignore_command:
            continue
          # else:
          #   print(f"{route} : {seq}")

          if estimate_class_distributions:
            with gzip.open(measurement[-1] + f'/{(seq):04}.json.gz', 'rt', encoding='utf-8') as f:
              measurements_i = ujson.load(f)

            target_speed_index, angle_index = self.get_indices_speed_angle(target_speed=measurements_i['target_speed'],
                                                                           brake=measurements_i['brake'],
                                                                           angle=measurements_i['angle'])

            self.angle_distribution.append(angle_index)
            self.speed_distribution.append(target_speed_index)
          if self.config.lidar_seq_len > 1:
            # load input seq and pred seq jointly
            temporal_lidar = []
            temporal_measurement = []
            for idx in range(self.config.lidar_seq_len):
              if not self.config.use_plant:
                assert self.config.seq_len == 1  # Temporal LiDARs are only supported with seq len 1 right now
                temporal_lidar.append(route_dir + '/lidar' + (f'/{(seq - idx):04}.laz'))
                temporal_measurement.append(route_dir + '/measurements' + (f'/{(seq - idx):04}.json.gz'))

            self.temporal_lidars.append(temporal_lidar)
            self.temporal_measurements.append(temporal_measurement)

          self.num_data += 1
          self.images.append(image)
          self.images_augmented.append(image_augmented)
          self.semantics.append(semantic)
          self.semantics_augmented.append(semantic_augmented)
          self.bev_semantics.append(bev_semantic)
          self.bev_semantics_augmented.append(bev_semantic_augmented)
          self.depth.append(depth)
          self.depth_augmented.append(depth_augmented)
          self.lidars.append(lidar)
          self.boxes.append(box)
          self.future_boxes.append(future_box)
          self.measurements.append(measurement)
          self.sample_start.append(seq)

    if estimate_class_distributions:
      classes_target_speeds = np.unique(self.speed_distribution)
      target_speed_weights = compute_class_weight(class_weight='balanced',
                                                  classes=classes_target_speeds,
                                                  y=self.speed_distribution)

      config.target_speed_weights = target_speed_weights.tolist()
      print('config.target_speeds: ', config.target_speeds)
      print('config.target_speed_bins: ', config.target_speed_bins)
      print('classes_target_speeds: ', classes_target_speeds)
      print('Target speed weights: ', config.target_speed_weights)
      unique, counts = np.unique(self.speed_distribution, return_counts=True)
      ts_dict = dict(zip(unique, counts))
      print('Target speed counts: ', ts_dict)
      with open(f'ts_dict{len(unique)}.pickle', 'wb') as handle:
        print('saving ts_dict')
        pickle.dump(ts_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

      classes_angles = np.unique(self.angle_distribution)
      angle_weights = compute_class_weight(class_weight='balanced', classes=classes_angles, y=self.angle_distribution)

      config.angle_weights = angle_weights.tolist()
      sys.exit()

    if estimate_sem_distribution:
      classes_semantic = np.unique(self.semantic_distribution)
      semantic_weights = compute_class_weight(class_weight='balanced',
                                              classes=classes_semantic,
                                              y=self.semantic_distribution)

      print('Semantic weights:', semantic_weights)

    del self.angle_distribution
    del self.speed_distribution
    del self.semantic_distribution

    # There is a complex "memory leak"/performance issue when using Python
    # objects like lists in a Dataloader that is loaded with
    # multiprocessing, num_workers > 0
    # A summary of that ongoing discussion can be found here
    # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
    # A workaround is to store the string lists as numpy byte objects
    # because they only have 1 refcount.
    self.images = np.array(self.images).astype(np.string_)
    self.images_augmented = np.array(self.images_augmented).astype(np.string_)
    self.semantics = np.array(self.semantics).astype(np.string_)
    self.semantics_augmented = np.array(self.semantics_augmented).astype(np.string_)
    self.bev_semantics = np.array(self.bev_semantics).astype(np.string_)
    self.bev_semantics_augmented = np.array(self.bev_semantics_augmented).astype(np.string_)
    self.depth = np.array(self.depth).astype(np.string_)
    self.depth_augmented = np.array(self.depth_augmented).astype(np.string_)
    self.lidars = np.array(self.lidars).astype(np.string_)
    self.boxes = np.array(self.boxes).astype(np.string_)
    self.future_boxes = np.array(self.future_boxes).astype(np.string_)
    self.measurements = np.array(self.measurements).astype(np.string_)

    self.temporal_lidars = np.array(self.temporal_lidars).astype(np.string_)
    self.temporal_measurements = np.array(self.temporal_measurements).astype(np.string_)
    self.sample_start = np.array(self.sample_start)
    if rank == 0:
      print(f'Loading {self.num_data} sample from {len(root)} folders')
      print('Total amount of routes:', total_routes)
      print('Skipped routes:', skipped_routes)
      print('Trainable routes:', trainable_routes)

    self.process_images = Process_Images(self.config, self.data_cache, self.images, self.images_augmented, self.validation)
    self.process_bev = Process_Bev(self.config, self.data_cache, self.bev_semantics, self.bev_semantics_augmented, self.validation)

  def __len__(self):
    """Returns the length of the dataset. """
    return self.num_data

  def __getitem__(self, index):
    """Returns the item at index idx. """
    # Disable threading because the data loader will already split in processes.
    cv2.setNumThreads(0)

    data = {}

    # Determine whether the augmented camera or the normal camera is used.
    if random.random() <= self.config.augment_percentage and self.config.augment:
      use_augment_sample = True
    else:
      use_augment_sample = False

    if not self.config.use_plant:
      if self.config.use_bev_semantic:
        data_bev, bev_postaug_rot = self.process_bev.process_data(index, use_augment_sample)
      data_images = self.process_images.process_data(index, use_augment_sample, bev_postaug_rot)

    data = data_images | data_bev

    return data

  def get_indices_speed_angle(self, target_speed, brake, angle):
    target_speed_index = np.digitize(x=target_speed, bins=self.target_speed_bins)

    # Define the first index to be the brake action
    if brake:
      target_speed_index = 0
    else:
      target_speed_index += 1

    angle_index = np.digitize(x=angle, bins=self.angle_bins)

    return target_speed_index, angle_index