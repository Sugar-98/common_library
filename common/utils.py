import numpy as np
import torch
import cv2
import os

class Camera:
  def __init__(self, trans, W, H,name, rot_quaternion = None, rot_euler = None,intrins = None, fov = None):
    """
    Define camera parameters. 
    Use right-handed coordinate system as a vehicle coordinate system, 
    and convert it to left-handed when attached to CARLA. 
    Origin is defined at the center of the vehicle. 
    Use nuScenes coordinate system as a camera coordinate system, 
    convered to CARLA coordinate system when attached as well. 
    """
    self.name = name
    self.trans = trans
    self.H = H
    self.W = W

    if rot_quaternion is not None:
      self.rot_quaternion = rot_quaternion
      self.rot_euler = self.quaternion2euler(rot_quaternion)
    elif rot_euler is not None:
      self.rot_euler = rot_euler
      self.rot_quaternion = self.euler2quaternion(rot_euler)
    else:
      self.rot_euler = None
      self.rot_quaternion = None
    
    if intrins is not None:
      self.intrins = intrins
      self.fov = self.intrinsic2fov(intrins, self.H, self.W)
    elif fov is not None:
      self.intrins = self.fov2intrins(fov, self.H, self.W)
      self.fov = fov
    else:
      self.intrins = None
      self.fov = None

  def set_rot_quaternion(self, rot_quaternion):
    self.rot_quaternion = rot_quaternion
    self.rot_euler = self.quaternion2euler(rot_quaternion)

  def set_rot_euler(self, rot_euler):
    self.rot_euler = rot_euler
    self.rot_quaternion = self.euler2quaternion(rot_euler)

  def set_intrins(self, intrins):
    self.intrins = intrins
    self.fov = self.intrinsic2fov(intrins, self.H, self.W)

  def set_fov(self, fov):
    self.intrins = self.fov2intrins(fov, self.H, self.W)
    self.fov = fov

  @staticmethod
  def quaternion2euler(rot_quaternion):
    w, x, y, z = rot_quaternion

    # 正規化
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        raise ValueError("Zero-norm quaternion cannot be converted to Euler angles.")
    w, x, y, z = w/n, x/n, y/n, z/n

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    rot_euler = [float(roll), float(pitch), float(yaw)]
    return rot_euler
  
  @staticmethod
  def euler2quaternion(rot_euler):
    roll, pitch, yaw = rot_euler

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # 正規化
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    rot_quaternion = [float(w), float(x), float(y), float(z)]
    return rot_quaternion
  
  @staticmethod
  def intrinsic2fov(intrins, H, W):
    K = np.array(intrins, dtype=float)
    fx = K[0, 0]  # fx from intrinsic

    fov = 2 * np.arctan(W / (2 * fx))
    return fov
  
  @staticmethod
  def fov2intrins(fov, H, W):
    fx = W / (2 * np.tan(fov / 2))
    fy = fx  # square pixels assumption
    cx = W / 2.0
    cy = H / 2.0

    intrins = [
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,   1]
    ]

    return intrins
  
  def offset_position(self, trans):
    self.trans[0] += trans[0]
    self.trans[1] += trans[1]
    self.trans[2] += trans[2]

  def get_rot_mat(self):
    w = self.rot_quaternion[0]
    x = self.rot_quaternion[1]
    y = self.rot_quaternion[2]
    z = self.rot_quaternion[3]

    #正規化
    n = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    rot_mat = [
      [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
      [ 2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
      [ 2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ]
    return rot_mat

def conv_NuScenes2Carla(trans_nu, rot_mat_nu):
  trans_carla = trans_nu
  trans_carla[1] = -trans_nu[1]

  rot_mat_nu = np.asarray(rot_mat_nu)
  R_veh2img = np.array([[0, 0, 1],
                        [-1, 0, 0],
                        [0, -1, 0]], dtype=float)
  
  rev_y_axis = np.diag([1.0, -1.0, 1.0])
  R_tmp = rot_mat_nu @ R_veh2img.T
  R_carla = rev_y_axis @ R_tmp @ rev_y_axis

  sy = -R_carla[2,0]
  pitch = np.arcsin(np.clip(sy, -1.0, 1.0))
  cp = np.cos(pitch)

  if cp > 1e-8:
    roll  = np.arctan2(R_carla[2,1], R_carla[2,2])
    yaw   = np.arctan2(R_carla[1,0], R_carla[0,0])
  else:
    roll  = 0.0
    yaw   = np.arctan2(-R_carla[0,1], R_carla[1,1])
  
  rot_carla = [roll, pitch, yaw]
  return trans_carla, rot_carla

def post_augument(img, data_aug_conf, augmentation):
  post_rot = np.eye(2, dtype=float)
  post_tran = np.zeros(2, dtype=float)
  # augmentation (resize, crop, horizontal flip, rotate)
  resize, resize_dims, crop, flip, rotate = define_augmentation(data_aug_conf, augmentation)
  img_augumented, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
         resize=resize,
         resize_dims=resize_dims,
         crop=crop,
         flip=flip,
         rotate=rotate,
         )
  
  # for convenience, make augmentation matrices 3x3
  post_tran = np.zeros(3, dtype=float)
  post_rot = np.eye(3, dtype=float)
  post_tran[:2] = post_tran2
  post_rot[:2, :2] = post_rot2

  return img_augumented, post_rot, post_tran

def define_augmentation(data_aug_conf, augmentation):
  H, W = data_aug_conf['H'], data_aug_conf['W']
  fH, fW = data_aug_conf['final_dim']

  if not augmentation:
    resize = max(fH/H, fW/W)
    resize_dims = (int(W*resize), int(H*resize))
    newW, newH = resize_dims
    crop_h = int((1 - np.mean(data_aug_conf['bot_pct_lim']))*newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    flip = False
    rotate = 0
  else:
    resize = np.random.uniform(*data_aug_conf['resize_lim'])
    resize_dims = (int(W*resize), int(H*resize))
    newW, newH = resize_dims
    crop_h = int((1 - np.random.uniform(*data_aug_conf['bot_pct_lim']))*newH) - fH
    crop_w = int(np.random.uniform(0, max(0, newW - fW)))
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    flip = False
    if data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
      flip = True
    rotate = np.random.uniform(*data_aug_conf['rot_lim'])

  return resize, resize_dims, crop, flip, rotate

def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    """
    img : 入力画像
    post_rot : eye(2)
    post_tran : zeros(2)
    resize : リサイズ比率
    resize_dims : リサイズ後
    crop : (左端のx, 上端のy, 右端のx, 下端のy)
    flip : 左右判定有無
    rotate : 
    """
    # adjust image
    #img = img.resize(resize_dims)
    img = cv2.resize(img, resize_dims, interpolation=cv2.INTER_LINEAR)
    #img = img.crop(crop)
    img = crop_image_with_pad(img, crop)
    if flip:
      #img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
      img = cv2.flip(img, 1)
    #img = img.rotate(rotate)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2.0, h/2.0), rotate, 1.0)  # 反時計回り
    img = cv2.warpAffine(img, M, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)

    # post-homography transformation
    post_rot *= resize
    post_tran -= np.array(crop[:2])
    if flip:
        A = np.array([[-1, 0], [0, 1]], dtype=float)
        b = np.array([crop[2] - crop[0], 0], dtype=float)
        post_rot = A @ post_rot
        post_tran = A @ post_tran + b
    A = get_rot(rotate/180*np.pi)
    b = np.array([crop[2] - crop[0], crop[3] - crop[1]], dtype=float) / 2.0
    b = A @ (-b) + b
    post_rot = A @ post_rot
    post_tran = A @ post_tran + b

    return img, post_rot, post_tran

def get_rot(h):
  return np.array([
    [np.cos(h), np.sin(h)],
    [-np.sin(h), np.cos(h)]
  ], dtype=float)

def get_rot3d (h):
  return np.array([
    [ np.cos(h), -np.sin(h), 0.0],
    [ np.sin(h),  np.cos(h), 0.0],
    [ 0.0,             0.0,          1.0]
  ], dtype=np.float32)

def crop_image_with_pad(img, crop):
  H,W,_ = img.shape
  padded_img = np.full((crop[3] - crop[1], crop[2] - crop[0], 3), 0, dtype=img.dtype)
  padded_img[0:min(H, crop[3] - crop[1]),0:min(W,crop[2] - crop[0])] = img[crop[1]:crop[3], crop[0]:min(W,crop[2])]
  return padded_img

def nomalize_image(img, mean, std):
  img = img/255.0
  mean = np.array(mean)
  std = np.array(std)

  img = (img - mean) / std
  return img

def bev_post_augument(bev, data_aug_conf):
  rotate = np.random.uniform(*data_aug_conf['bev_rot_lim'])
  h, w = bev.shape[:2]
  rot_mat = cv2.getRotationMatrix2D((w/2.0, h/2.0), rotate, 1.0)  # 反時計回り
  bev_augmented = cv2.warpAffine(bev, rot_mat, (w, h),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0)
  return bev_augmented, rotate

def rot_cam_coordi_sys(trans, rots, rotate):
  rotmat = get_rot3d(rotate/180*np.pi)
  trans = (rotmat @ trans.T).T
  rots = np.einsum('ij, njk -> nik', rotmat, rots)
  return trans, rots

#def get_bev_semantic_rgb(bev_semantic, bev_classes_list):
#  converter = np.array(bev_classes_list)
#  converter[1][0:3] = 40
#  bev_semantic_rgb = converter[bev_semantic, ...].astype('uint8')
#  return bev_semantic_rgb

def get_bev_semantic_rgb(probs_bev, bev_classes_list):
    """
    probs_bev: np.ndarray [C,H,W]  各クラスの確率 (softmax後)
    bev_classes_list: list[(R,G,B)]  各クラスの色定義
    """
    probs_bev_T = np.rot90(probs_bev, k=1, axes=(1, 2))

    C, H, W = probs_bev_T.shape
    colors = np.array(bev_classes_list)[:, :3]  # [C,3]

    # 各ピクセルのRGBを確率加重平均
    # → 各クラスの確率 * 色 を足し合わせる
    bev_rgb = np.tensordot(probs_bev_T.transpose(1,2,0), colors, axes=([2],[0]))
    bev_rgb = np.clip(bev_rgb, 0, 255).astype(np.uint8)
    return bev_rgb

def cal_mfb_weights(
    hist, 
    ignore_class=None,
    eps=1e-6,
    max_weight=3.0,
    min_class0_weight=0.5,
    normalize=True):
    mask = np.ones_like(hist, dtype=bool)
    mask[ignore_class] = False
    p = hist[mask] / (hist[mask].sum() + eps)
    median_p = np.median(p[p > 0])
    w_valid = median_p / (p + eps)
    w_valid = np.clip(w_valid, a_min=None, a_max=max_weight)
    if normalize:
      w_valid = w_valid / w_valid.mean()
    w = np.zeros_like(hist, dtype=np.float32)
    w[mask] = w_valid
    w[ignore_class] = 0.0
    w[0] = max(w[0].item(), min_class0_weight)
    return w.tolist()

def compute_tp_fp_fn_multiclass(logits, label, num_classes, ignore_class=[]):
    pred = logits.argmax(dim=1)   # (B, H, W)
    pred = pred.view(-1)
    label = label.view(-1)

    ignore_mask = torch.zeros_like(label, dtype=torch.bool)
    for cls in ignore_class:
        ignore_mask |= (label == cls)

    valid_mask = ~ignore_mask

    pred = pred[valid_mask]
    label = label[valid_mask]

    tp = torch.zeros(num_classes, dtype=torch.long)
    fp = torch.zeros(num_classes, dtype=torch.long)
    fn = torch.zeros(num_classes, dtype=torch.long)

    for c in range(num_classes):

        if c in ignore_class:
            continue

        tp[c] = ((pred == c) & (label == c)).sum()
        fp[c] = ((pred == c) & (label != c)).sum()
        fn[c] = ((pred != c) & (label == c)).sum()

    return  tp,fp,fn

class MP4Writer:
  def __init__(self, save_path, fps=10):
    self.save_path = save_path
    self.fps = fps
    self.writer = None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

  def write(self, img_rgb):
    """
    img_rgb: np.ndarray (H, W, 3), uint8, RGB
    """
    h, w, _ = img_rgb.shape

    if self.writer is None:
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      self.writer = cv2.VideoWriter(
        self.save_path, fourcc, self.fps, (w, h)
      )

    img_bgr = img_rgb[..., ::-1]  # RGB -> BGR
    self.writer.write(img_bgr)

  def close(self):
    if self.writer is not None:
      self.writer.release()
      self.writer = None