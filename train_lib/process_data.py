import cv2
import numpy as np
from imgaug import augmenters as ia
from common.utils import post_augument, nomalize_image, bev_post_augument, rot_cam_coordi_sys

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

      if self.config.use_post_augment:
        img, post_rot, post_tran = post_augument(img, self.config.data_aug_conf, self.validation)
      else:
        img, post_rot, post_tran = post_augument(img, self.config.data_aug_conf, augmentation=True)

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
    data["intrins"] = np.array(intrins)
    data["rots"] = rots
    data["trans"] = trans
    data["post_rots"] = np.array(post_rots)
    data["post_trans"] = np.array(post_trans)
    
    # The transpose change the image into pytorch (C,H,W) format
    data['rgb'] = np.transpose(imgs[0], (2, 0, 1))

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
  
  def process_measurements(self, index, augment_sample):
    data = {}
    return data
  
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