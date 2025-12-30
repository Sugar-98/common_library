from common.utils import Camera
from common.config.Agent_conf import Agent_conf
import numpy as np

class DataAgent_conf(Agent_conf):
  def __init__(self):
    # -----------------------------------------------------------------------------
    # DataAgent
    # -----------------------------------------------------------------------------
    self.augment = 0  # Whether to use rotation and translation augmentation
    # Max and min values by which the augmented camera is shifted left and right
    self.camera_translation_augmentation_min = -1.0
    self.camera_translation_augmentation_max = 1.0
    # Max and min values by which the augmented camera is rotated around the yaw
    # Numbers are in degree
    self.camera_rotation_augmentation_min = -5.0
    self.camera_rotation_augmentation_max = 5.0
    # Every data_save_freq frame the data is stored during training
    # Set to one for backwards compatibility. Released dataset was collected with 5
    self.data_save_freq = 5
    # LiDAR compression parameters
    self.point_format = 0  # LARS point format used for storing
    self.point_precision = 0.01  # Precision up to which LiDAR points are stored
    
    self.conf_enable_disturbance = False #Add steering disturbance for agent
    self.conf_plot_control = False #Plot control signals during data collection
    # -----------------------------------------------------------------------------
    # Sensor config
    # -----------------------------------------------------------------------------
    self.gen_bev_semantics = True
    self.gen_boxes = False
    self.gen_depth = False
    self.gen_lidar = False
    self.gen_rgb = True
    self.gen_semantics = False

    self.lidar_pos = [0.0, 0.0, 2.5]  # x, y, z mounting position of the LiDAR
    self.lidar_rot = [0.0, 0.0, -90.0]  # Roll Pitch Yaw of LiDAR in degree
    self.lidar_rotation_frequency = 10  # Number of Hz at which the Lidar operates
    # Number of points the LiDAR generates per second.
    # Change in proportion to the rotation frequency.
    self.lidar_points_per_second = 600000

    # Width and height of the LiDAR grid that the point cloud is voxelized into.
    self.lidar_resolution_width = 256
    self.lidar_resolution_height = 256

    # Pixels per meter used in the semantic segmentation map during data collection.
    # On Town 13 2.0 is the highest that opencv can handle.
    self.pixels_per_meter_collection = 2.0

    # Therefore their size is smaller
    self.camera_width = 1600  # Camera width in pixel during data collection and eval (affects sensor agent)
    self.camera_height = 900  # Camera height in pixel during data collection and eval (affects sensor agent)
    origin2rear = [-1.4, 0, 0.25] #vehicle.lincoln.mkz_2017

    tmp_trans = [1.70079118954, 0.0159456324149, 1.51095763913]
    tmp_rot = [0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755]
    tmp_intrinsic = [
      [1266.417203046554, 0.0, 816.2670197447984],
      [0.0, 1266.417203046554, 491.50706579294757],
      [0.0, 0.0, 1.0]
    ]
    self.CAM_FRONT = Camera(tmp_trans, self.camera_width, self.camera_height, name="CAM_FRONT",\
                            rot_quaternion = tmp_rot, intrins = tmp_intrinsic)

    tmp_trans = [0.0283260309358, 0.00345136761476, 1.57910346144]
    tmp_rot = [0.5037872666382278, -0.49740249788611096, -0.4941850223835201, 0.5045496097725578]
    tmp_intrinsic = [
      [809.2209905677063,0.0,829.2196003259838],
      [0.0,809.2209905677063,481.77842384512485],
      [0.0,0.0,1.0]
    ]
    self.CAM_BACK = Camera(tmp_trans, self.camera_width, self.camera_height, name="CAM_BACK",\
                            rot_quaternion = tmp_rot, intrins = tmp_intrinsic)
    
    tmp_trans = [1.03569100218,0.484795032713,1.59097014818]
    tmp_rot = [0.6924185592174665,-0.7031619420114925,-0.11648342771943819,0.11203317912370753]
    tmp_intrinsic = [
      [1256.7414812095406,0.0,792.1125740759628],
      [0.0,1256.7414812095406,492.7757465151356],
      [0.0,0.0,1.0]
    ]
    self.CAM_BACK_LEFT = Camera(tmp_trans, self.camera_width, self.camera_height, name="CAM_BACK_LEFT",\
                            rot_quaternion = tmp_rot, intrins = tmp_intrinsic)
    
    tmp_trans = [1.52387798135,0.494631336551,1.50932822144]
    tmp_rot = [0.6757265034669446,-0.6736266522251881,0.21214015046209478,-0.21122827103904068]
    tmp_intrinsic = [
      [1272.5979470598488,0.0,826.6154927353808],
      [0.0,1272.5979470598488,479.75165386361925],
      [0.0,0.0,1.0]
    ]
    self.CAM_FRONT_LEFT = Camera(tmp_trans, self.camera_width, self.camera_height, name="CAM_FRONT_LEFT",\
                            rot_quaternion = tmp_rot, intrins = tmp_intrinsic)
    
    tmp_trans = [1.5508477543,-0.493404796419,1.49574800619]
    tmp_rot = [0.2060347966337182,-0.2026940577919598,0.6824507824531167,-0.6713610884174485]
    tmp_intrinsic = [
      [1260.8474446004698,0.0,807.968244525554],
      [0.0,1260.8474446004698,495.3344268742088],
      [0.0,0.0,1.0]
    ]
    self.CAM_FRONT_RIGHT = Camera(tmp_trans, self.camera_width, self.camera_height, name="CAM_FRONT_RIGHT",\
                            rot_quaternion = tmp_rot, intrins = tmp_intrinsic)

    tmp_trans = [1.0148780988,-0.480568219723,1.56239545128]
    tmp_rot = [0.12280980120078765,-0.132400842670559,-0.7004305821388234,0.690496031265798]
    tmp_intrinsic = [
      [1259.5137405846733,0.0,807.2529053838625],
      [0.0,1259.5137405846733,501.19579884916527],
      [0.0,0.0,1.0]
    ]
    self.CAM_BACK_RIGHT = Camera(tmp_trans, self.camera_width, self.camera_height, name="CAM_BACK_RIGHT",\
                            rot_quaternion = tmp_rot, intrins = tmp_intrinsic)
    
    self.cameras = [
      self.CAM_FRONT,
      self.CAM_BACK,
      self.CAM_BACK_LEFT,
      self.CAM_FRONT_LEFT,
      self.CAM_FRONT_RIGHT,
      self.CAM_BACK_RIGHT
    ]
    self.num_cameras = 0
    for idx, camera in enumerate(self.cameras):
      camera.offset_position(origin2rear)
      self.num_cameras += 1

    # Crop the image during training to the values below. also affects the transformer tokens self.img_vert_anchors
    self.crop_image = True
    # self.cropped_height = 384  # crops off the bottom part
    # self.cropped_width = 1024  # crops off both sides symmetrically
    self.cropped_height = 384  # crops off the bottom part
    self.cropped_width = 1024  # crops off both sides symmetrically

    # -----------------------------------------------------------------------------
    # Kinematic Bicycle Model
    # -----------------------------------------------------------------------------
    #  Time step for the model (20 frames per second).
    self.time_step = 1. / 20.
    # Kinematic bicycle model parameters tuned from World on Rails.
    # Distance from the rear axle to the front axle of the vehicle.
    self.front_wheel_base = -0.090769015
    # Distance from the rear axle to the center of the rear wheels.
    self.rear_wheel_base = 1.4178275
    # Gain factor for steering angle to wheel angle conversion.
    self.steering_gain = 0.36848336
    # Deceleration rate when braking (m/s^2) of other vehicles.
    self.brake_acceleration = -4.952399
    # Acceleration rate when throttling (m/s^2) of other vehicles.
    self.throttle_acceleration = 0.5633837
    # Tuned parameters for the polynomial equations modeling speed changes
    # Numbers are tuned parameters for the polynomial equations below using
    # a dataset where the car drives on a straight highway, accelerates to
    # and brakes again
    # Coefficients for polynomial equation estimating speed change with throttle input for ego model.
    self.throttle_values = np.array([
        9.63873001e-01, 4.37535692e-04, -3.80192912e-01, 1.74950069e+00, 9.16787414e-02, -7.05461530e-02,
        -1.05996152e-03, 6.71079346e-04
    ])
    # Coefficients for polynomial equation estimating speed change with brake input for the ego model.
    self.brake_values = np.array([
        9.31711370e-03, 8.20967431e-02, -2.83832427e-03, 5.06587474e-05, -4.90357228e-07, 2.44419284e-09,
        -4.91381935e-12
    ])
    # Minimum throttle value that has an affect during forecasting the ego vehicle.
    self.throttle_threshold_during_forecasting = 0.3