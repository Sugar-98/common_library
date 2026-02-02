class Agent_conf:
  def __init__(self):
    """
    Common settings for all agent
    """
    # -----------------------------------------------------------------------------
    # Agent file
    # -----------------------------------------------------------------------------
    self.carla_frame_rate = 1.0 / 20.0  # CARLA frame rate in milliseconds
    # Iou threshold used for non-maximum suppression on the Bounding Box
    # predictions for the ensembles
    self.iou_treshold_nms = 0.2
    self.route_planner_min_distance = 7.5
    self.route_planner_max_distance = 50.0
    # Min distance to the waypoint in the dense rout that the expert is trying to follow
    self.dense_route_planner_min_distance = 2.4
    # Number of frames after which the creep controller starts triggering. 1100 is larger than wait time at red light.
    self.stuck_threshold = 1100
    self.creep_duration = 20  # Number of frames we will creep forward
    self.creep_throttle = 0.4
    # CARLA needs some time to initialize in which the cars actions are blocked.
    # Number tuned empirically
    self.inital_frames_delay = 2.0 / self.carla_frame_rate
    self.slower_factor = 0.8  # Factor by which the target speed will be reduced during inference if slower is active

    # Extent of the ego vehicles bounding box
    self.ego_extent_x = 2.4508416652679443
    self.ego_extent_y = 1.0641621351242065
    self.ego_extent_z = 0.7553732395172119

    # Size of the safety box
    self.safety_box_z_min = 0.5
    self.safety_box_z_max = 1.5

    self.safety_box_y_min = -self.ego_extent_y * 0.8
    self.safety_box_y_max = self.ego_extent_y * 0.8

    self.safety_box_x_min = self.ego_extent_x
    self.safety_box_x_max = self.ego_extent_x + 2.5

    # Probability 0 - 1. If the confidence in the brake action is higher than this
    # value brake is chosen as the action.
    self.brake_uncertainty_threshold = 0.9  # 1 means that it is not used at all