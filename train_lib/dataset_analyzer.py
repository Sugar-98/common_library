import torch
from tqdm import tqdm
import os, json
import numpy as np
import matplotlib.pyplot as plt

class Analyzer:
  def __init__(self,
          dataloader_train,
          dataloader_val,
          num_train,
          num_val,
          calra_config,
          analysis_config=None):
    self.dataloader_train = dataloader_train
    self.dataloader_val = dataloader_val
    self.calra_config = calra_config
    self.num_train = num_train
    self.num_val = num_val
    self.analysis_config = analysis_config

  def __call__(self, save_path = None):
    seq_data, statistics_data = self.get_data_analysis()
    if save_path is not None:
      self.save_data(save_path, seq_data, statistics_data)
    self.plot_analysis(seq_data, statistics_data)

    return seq_data, statistics_data
    
  def plot_analysis(self, seq_data, statistics_data):
    """
    Generic plotter for statistics_data.

    Convention:
    - Keys are formatted as "<base>_<suffix>" (split by the last underscore).
    - Entries with the same <base> are plotted on the same figure.
    - Legend uses <suffix>, and title uses <base>.
    """
    # group by base key
    groups = {}
    for k, v in statistics_data.items():
      if "_" not in k:
        base, suffix = k, "default"
      else:
        base, suffix = k.rsplit("_", 1)  # last '_' only
      groups.setdefault(base, []).append((suffix, v))

    for base, series in groups.items():
        # Convert to 1D numeric arrays when possible
        converted = []
        for suffix, v in series:
          if hasattr(v, "detach"):  # torch tensor
            v = v.detach().cpu().numpy()
          # Skip dict values (e.g. town_counts) -- handled separately
          if isinstance(v, dict):
            self._plot_dict_bar(base, suffix, v)
            continue
          arr = np.asarray(v)
          # Skip non-numeric or multi-dim data (keep it simple)
          if not np.issubdtype(arr.dtype, np.number):
              continue
          if arr.ndim == 0:
              arr = arr.reshape(1)
          if arr.ndim != 1:
              continue
          converted.append((suffix, arr.astype(np.float64)))

        if not converted:
          continue

        # Align lengths: only plot if all series have the same length
        lens = [value.shape[0] for _, value in converted]
        if len(set(lens)) != 1:
          # fallback: plot each as a line with its own x
          plt.figure()
          for suffix, value in converted:
            plt.plot(np.arange(len(value)), value, label=suffix)
          plt.title(base)
          plt.xlabel("index")
          plt.ylabel("value")
          plt.legend()
          continue

        num_idx = lens[0]
        idx = np.arange(num_idx)
        m = len(converted)
        width = 0.8 / max(m, 1)

        plt.figure()
        for i, (suffix, value) in enumerate(sorted(converted, key=lambda t: t[0])):
          plt.bar(idx - 0.4 + width/2 + i*width, value, width, label=suffix)

        # Use bin edges as x-tick labels for histogram-style plots
        base_prefix = base.rsplit('_', 1)[0] if base.endswith(('counts', 'rate')) else base
        possible_bin_key = f"{base_prefix}_bins"
        if possible_bin_key in statistics_data:
          edges = statistics_data[possible_bin_key]
          if len(edges) == num_idx + 1:
            tick_labels = [f"{edges[j]:.2f}" for j in range(num_idx)]
            plt.xticks(idx, tick_labels, rotation=45, fontsize=7)

        # Use command labels if applicable
        if 'command' in base and self.analysis_config and hasattr(self.analysis_config, 'command_labels'):
          labels = self.analysis_config.command_labels
          if len(labels) == num_idx:
            plt.xticks(idx, labels, rotation=30, fontsize=8)

        # Use '0' / '1' labels for binary fields
        if any(bname in base for bname in ['brake', 'junction']) and num_idx == 2:
          plt.xticks(idx, ['0', '1'])

        plt.title(base)
        plt.xlabel("index")
        plt.ylabel("value")
        plt.legend()

    plt.show()

  @staticmethod
  def _plot_dict_bar(base, suffix, d):
    """Plot a dict as a bar chart (used for town distributions)."""
    keys = list(d.keys())
    vals = [float(v) for v in d.values()]
    plt.figure()
    plt.bar(range(len(keys)), vals, label=suffix)
    plt.xticks(range(len(keys)), keys)
    plt.title(base)
    plt.xlabel("category")
    plt.ylabel("value")
    plt.legend()

  def save_data(self, save_path, seq_data, statistics_data):
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'statistics_data.json'), 'w') as fjson:
      json.dump(statistics_data, fjson, indent=2)

    # Save analysis config as JSON
    if self.analysis_config is not None:
      config_dict = {k: v for k, v in vars(self.analysis_config).items() if not k.startswith('_')}
      for k, v in config_dict.items():
        if isinstance(v, np.ndarray):
          config_dict[k] = v.tolist()
      with open(os.path.join(save_path, 'analysis_config.json'), 'w') as fjson:
        json.dump(config_dict, fjson, indent=2)
    
  def get_data_analysis(self):
    """
    Run analysis on both train and validation loaders and merge the results.
    """
    seq_data_train, statistics_data_train = self.analyze_data(self.dataloader_train, self.num_train, suffix = "_train")
    seq_data_val, statistics_data_val = self.analyze_data(self.dataloader_val, self.num_val, suffix = "_val")
    seq_data = seq_data_train | seq_data_val
    statistics_data = statistics_data_train | statistics_data_val

    return seq_data, statistics_data

  def analyze_data(self, dataloader, num_data, suffix = ""):
    """
    Extract dataset-level statistics from a dataloader.

    The returned dictionaries are designed to be extensible:
    - seq_data: for per-sample / sequential outputs (optional)
    - statistics_data: for aggregated summary metrics
    """
    num_bev_classes = len(self.calra_config.bev_converter)
    hist_bev = torch.zeros(num_bev_classes, dtype=torch.long)
    bev_class_presence = torch.zeros(num_bev_classes, dtype=torch.long)

    # Measurement distribution accumulators
    all_steer = []
    all_speed = []
    all_throttle = []
    all_theta = []
    all_brake = []
    all_junction = []
    all_command = []
    all_town = []

    with torch.no_grad():
      for data in tqdm(dataloader):
        # BEV analysis
        bev_semantics = data["bev_semantic"]
        hist_bev_tmp = torch.bincount(bev_semantics[:,::4,::4].flatten(), minlength=num_bev_classes)
        hist_bev += hist_bev_tmp

        for c in range(num_bev_classes):
          bev_class_presence[c] += (bev_semantics == c).any(dim=(1,2)).sum()

        # Measurement collection
        all_steer.append(data['steer'].cpu().numpy().flatten())
        all_speed.append(data['speed'].cpu().numpy().flatten())
        all_throttle.append(data['throttle'].cpu().numpy().flatten())
        all_theta.append(data['theta'].cpu().numpy().flatten())
        all_brake.append(data['brake'].cpu().numpy().flatten())
        all_junction.append(data['junction'].cpu().numpy().flatten())
        all_command.append(torch.argmax(data['command'], dim=1).cpu().numpy().flatten())
        all_town.append(data['town'].cpu().numpy().flatten())

    pixel_total = hist_bev.sum().clamp(min=1)
    bev_pixel_rate = (hist_bev.float() / pixel_total).tolist()
    bev_class_presence_rate = (bev_class_presence.float()/num_data).tolist()

    # Build measurement histograms
    all_steer = np.concatenate(all_steer)
    all_speed = np.concatenate(all_speed)
    all_throttle = np.concatenate(all_throttle)
    all_theta = np.concatenate(all_theta)
    all_brake = np.concatenate(all_brake)
    all_junction = np.concatenate(all_junction)
    all_command = np.concatenate(all_command)
    all_town = np.concatenate(all_town)

    measurement_stats = self._compute_measurement_histograms(
        all_steer, all_speed, all_throttle, all_theta,
        all_brake, all_junction, all_command, all_town, suffix)

    seq_data = {}
    statistics_data = {
      f"Bev_class_presence_rate{suffix}": bev_class_presence_rate,
      f"Bev_pixel_rate{suffix}": bev_pixel_rate,
      **measurement_stats,
    }
    return seq_data, statistics_data

  def _compute_measurement_histograms(self, steer, speed, throttle, theta,
                      brake, junction, command, town, suffix):
    """
    Compute histogram distributions for measurement fields.
    Returns a dict of {key: list_of_counts_or_rates}.
    """
    stats = {}
    cfg = self.analysis_config

    # --- Continuous variables: use np.histogram with config bins ---
    if cfg is not None:
      steer_bins = cfg.steer_bins
      speed_bins = cfg.speed_bins
      throttle_bins = cfg.throttle_bins
      theta_bins = cfg.theta_bins
    else:
      steer_bins = np.linspace(-1.0, 1.0, 21).tolist()
      speed_bins = np.linspace(0.0, 25.0, 26).tolist()
      throttle_bins = np.linspace(0.0, 1.0, 11).tolist()
      theta_bins = np.linspace(-np.pi, np.pi, 25).tolist()

    for name, values, bins in [
        ('steer', steer, steer_bins),
        ('speed', speed, speed_bins),
        ('throttle', throttle, throttle_bins),
        ('theta', theta, theta_bins),
    ]:
      counts, edges = np.histogram(values, bins=bins)
      total = counts.sum()
      rate = (counts / max(total, 1)).tolist()
      stats[f"{name}_counts{suffix}"] = counts.tolist()
      stats[f"{name}_rate{suffix}"] = rate
      stats[f"{name}_bins"] = [float(e) for e in edges]

    # --- Binary variables: count 0 vs 1 ---
    for name, values in [('brake', brake), ('junction', junction)]:
      n_zero = int((values == 0).sum())
      n_one = int((values != 0).sum())
      total = n_zero + n_one
      stats[f"{name}_counts{suffix}"] = [n_zero, n_one]
      stats[f"{name}_rate{suffix}"] = [n_zero / max(total, 1), n_one / max(total, 1)]

    # --- Command: categorical [0..5] ---
    cmd_counts = np.bincount(command.astype(int), minlength=6)[:6]
    total = cmd_counts.sum()
    stats[f"command_counts{suffix}"] = cmd_counts.tolist()
    stats[f"command_rate{suffix}"] = (cmd_counts / max(total, 1)).tolist()

    # --- Town: categorical ---
    town_int = town.astype(int)
    unique_towns = np.unique(town_int)
    town_counts = {str(t): int((town_int == t).sum()) for t in unique_towns}
    total = sum(town_counts.values())
    town_rate = {k: v / max(total, 1) for k, v in town_counts.items()}
    stats[f"town_counts{suffix}"] = town_counts
    stats[f"town_rate{suffix}"] = town_rate

    return stats

