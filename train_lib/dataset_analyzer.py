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
          calra_config):
    self.dataloader_train = dataloader_train
    self.dataloader_val = dataloader_val
    self.calra_config = calra_config
    self.num_train = num_train
    self.num_val = num_val

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
          arr = np.asarray(v)
          # Skip non-numeric or multi-dim data (keep it simple)
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

        plt.title(base)
        plt.xlabel("index")
        plt.ylabel("value")
        plt.legend()

    plt.show()

  def save_data(self, save_path, seq_data, statistics_data):
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'statistics_data.json'), 'w') as fjson:
      json.dump(statistics_data, fjson, indent=2)
    
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
    steer = []
    speed = []
    command = []
    throttle = []
    brake = []

    num_bev_classes = len(self.calra_config.bev_converter)
    hist_bev = torch.zeros(num_bev_classes, dtype=torch.long)
    bev_class_presence = torch.zeros(num_bev_classes, dtype=torch.long)

    with torch.no_grad():
      for data in tqdm(dataloader):
        #steer.append(data['steer'].cpu().numpy())
        #speed.append(data['speed'].cpu().numpy())
        #command.append(torch.argmax(data['command'], dim=1).cpu().numpy())
        #throttle.append(data['throttle'].cpu().numpy())
        #brake.append(data['brake'].cpu().numpy())

        bev_semantics = data["bev_semantic"]
        hist_bev_tmp = torch.bincount(bev_semantics[:,::4,::4].flatten(), minlength=num_bev_classes)
        hist_bev += hist_bev_tmp

        for c in range(num_bev_classes):
          bev_class_presence[c] += (bev_semantics == c).any(dim=(1,2)).sum()

    pixel_total = hist_bev.sum().clamp(min=1)
    bev_pixel_rate = (hist_bev.float() / pixel_total).tolist()

    bev_class_presence_rate = (bev_class_presence.float()/num_data).tolist()
    
    seq_data = {}
    statistics_data = {
      f"Bev_class_presence_rate{suffix}":bev_class_presence_rate,
      f"Bev_pixel_rate{suffix}":bev_pixel_rate
    }
    return seq_data, statistics_data

