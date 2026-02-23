import numpy as np

class DataAnalysis_conf:
  def __init__(self):
    # --- Histogram bin edges for each measurement field ---
    # steer: normalized in [-1, 1]
    self.steer_bins = np.linspace(-1.0, 1.0, 21).tolist()
    # throttle: [0, 1]
    self.throttle_bins = np.linspace(0.0, 1.0, 11).tolist()
    # speed: m/s, typically [0, 25]
    self.speed_bins = np.linspace(0.0, 25.0, 26).tolist()
    # theta (yaw): radians, [-pi, pi]
    self.theta_bins = np.linspace(-np.pi, np.pi, 25).tolist()
    # brake: binary {0, 1} -- no bins needed, counted directly
    # junction: binary {0, 1} -- no bins needed, counted directly
    # town: integer categorical -- counted directly
    # command: 6-dim one-hot -> argmax index [0..5] -- counted directly
    self.command_labels = [
      'Left', 'Right', 'Straight', 'LaneFollow', 'ChangeLaneLeft', 'ChangeLaneRight'
    ]
