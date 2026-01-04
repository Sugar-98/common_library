
import torch
from tqdm import tqdm
import jsonpickle
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from itertools import zip_longest

class Engine:
  def __init__(self,
         model_wrapper,
         optimizer,
         dataloader_train,
         dataloader_val,
         Train_config,
         device,
         cur_epoch=0):
    self.cur_epoch = cur_epoch
    self.train_loss = []
    self.val_loss = []
    self.model_wrapper = model_wrapper
    self.optimizer = optimizer
    self.dataloader_train = dataloader_train
    self.dataloader_val = dataloader_val
    self.Train_config = Train_config
    self.device = device
    self.step = 0
  
  def train(self):
    self.model_wrapper.train()
    torch.autograd.set_detect_anomaly(True) #detect invalid grad
    num_batches = 0
    loss_epoch = 0.0
    self.model_wrapper.init_metrics()

    for i, data in enumerate(tqdm(self.dataloader_train)):
      self.optimizer.zero_grad()
      pred, label, loss, tmp_loss_individual = self.model_wrapper.load_data_compute_loss(data)
      loss.backward()

      if self.Train_config.use_grad_clip:
        # Since the gradients of optimizers assigned params are now unscaled, we can clip as usual.
        torch.nn.utils.clip_grad_norm_(self.model_wrapper.parameters(),
                      max_norm=int(self.Train_config.grad_clip_max_norm),
                      error_if_nonfinite=True)
        
      self.optimizer.step()
      num_batches += 1
      loss_epoch += float(loss.item())
      self.model_wrapper.cal_metrics_batch()

    metrics = self.model_wrapper.cal_metrics_epoch()
    print(f'-----------Loss_Train {loss_epoch/len(self.dataloader_train)}----------')
    self.model_wrapper.plot_model_out()

    if self.cur_epoch == 0:
      self.train_loss = []
      self.train_loss.append(loss_epoch/len(self.dataloader_train))

      self.train_metrics = {}
      for key, value in metrics.items():
        self.train_metrics[key] = []
        self.train_metrics[key].append(value)
    else:
      self.train_loss.append(loss_epoch/len(self.dataloader_train))

      for key, value in metrics.items():
        self.train_metrics[key].append(value)

  def validate(self):
    self.model_wrapper.eval()
    self.model_wrapper.init_metrics()

    num_batches = 0
    loss_epoch = 0.0
    if len(self.dataloader_val) != 0:
      with torch.no_grad():
        for data in tqdm(self.dataloader_val):
          pred, label, loss, tmp_loss_individual = self.model_wrapper.load_data_compute_loss(data)
          num_batches += 1
          loss_epoch += float(loss.item())
          self.model_wrapper.cal_metrics_batch()

        metrics = self.model_wrapper.cal_metrics_epoch()
        print(f'-----------Loss_Validate {loss_epoch/len(self.dataloader_val)}----------')
        self.model_wrapper.plot_model_out()

      if self.cur_epoch == 0:
        self.val_loss = []
        self.val_loss.append(loss_epoch/len(self.dataloader_val))
        
        self.val_metrics = {}
        for key, value in metrics.items():
          self.val_metrics[key] = []
          self.val_metrics[key].append(value)
      else:
        self.val_loss.append(loss_epoch/len(self.dataloader_val))

        for key, value in metrics.items():
          self.val_metrics[key].append(value)

  def plot_data(self):
    steer = torch.tensor([])
    speed = torch.tensor([])
    command = torch.tensor([])
    throttle = torch.tensor([])
    brake = torch.tensor([])
    for i, data in enumerate(tqdm(self.dataloader_train)):
      steer = torch.cat((steer, data['steer']))
      speed = torch.cat((speed, data['speed']))
      tmp_command = torch.argmax(data['command'], dim=1)
      command = torch.cat((command, tmp_command))
      throttle = torch.cat((throttle, data['throttle']))
      brake = torch.cat((brake, data['brake']))

    if not hasattr(self, 'ax_train'):
      fig_train, ax_train = plt.subplots(3, 3, num="Train data set")
    ax_train[0][0].hist(steer.tolist(), bins=100, range=(-1, 1))
    ax_train[0][0].set_xlabel("Steering value")
    ax_train[0][0].set_ylabel("Frequency")
    ax_train[0][0].grid(True)

    ax_train[1][0].hist(steer.tolist(), bins=100, range=(-0.1, 0.1))
    ax_train[1][0].set_xlabel("Steering value(low)")
    ax_train[1][0].set_ylabel("Frequency")
    ax_train[1][0].grid(True)

    ax_train[2][0].hist(steer.tolist(), bins=100, range=(-0.025, 0.025))
    ax_train[2][0].set_xlabel("Steering value(low)")
    ax_train[2][0].set_ylabel("Frequency")
    ax_train[2][0].grid(True)

    ax_train[0][1].hist(speed.tolist(), bins=100, range=(-20*1000/3600, 100*1000/3600))
    ax_train[0][1].set_xlabel("speed value")
    ax_train[0][1].set_ylabel("Frequency")
    ax_train[0][1].grid(True)

    ax_train[1][1].scatter(steer.tolist(), speed.tolist(), s=5)
    ax_train[1][1].set_xlabel("steer")
    ax_train[1][1].set_ylabel("speed")
    ax_train[1][1].grid(True)

    ax_train[2][1].hist(command.tolist(), bins=8, range=(-1, 7))
    ax_train[2][1].set_xlabel("command value")
    ax_train[2][1].set_ylabel("Frequency")
    ax_train[2][1].grid(True)

    ax_train[0][2].hist(throttle.tolist(), bins=100, range=(-1, 1))
    ax_train[0][2].set_xlabel("throttle")
    ax_train[0][2].set_ylabel("Frequency")
    ax_train[0][2].grid(True)

    ax_train[1][2].hist(brake.tolist(), bins=100, range=(-1, 1))
    ax_train[1][2].set_xlabel("brake")
    ax_train[1][2].set_ylabel("Frequency")
    ax_train[1][2].grid(True)

    steer_val = torch.tensor([])
    speed_val = torch.tensor([])
    command_val = torch.tensor([])
    throttle_val = torch.tensor([])
    brake_val = torch.tensor([])
    for i, data in enumerate(tqdm(self.dataloader_val)):
      steer_val = torch.cat((steer_val, data['steer']))
      speed_val = torch.cat((speed_val, data['speed']))
      tmp_command_val = torch.argmax(data['command'], dim=1)
      command_val = torch.cat((command_val, tmp_command_val))
      throttle_val = torch.cat((throttle_val, data['throttle']))
      brake_val = torch.cat((brake_val, data['brake']))
    
    if not hasattr(self, 'ax_val'):
      fig_val, ax_val = plt.subplots(3, 3, num="Validation data set")
    ax_val[0][0].hist(steer_val.tolist(), bins=100, range=(-1, 1))
    ax_val[0][0].set_xlabel("Steering value")
    ax_val[0][0].set_ylabel("Frequency")
    ax_val[0][0].grid(True)

    ax_val[1][0].hist(steer_val.tolist(), bins=100, range=(-0.1, 0.1))
    ax_val[1][0].set_xlabel("Steering value(low)")
    ax_val[1][0].set_ylabel("Frequency")
    ax_val[1][0].grid(True)

    ax_val[2][0].hist(steer_val.tolist(), bins=100, range=(-0.025, 0.025))
    ax_val[2][0].set_xlabel("Steering value(low)")
    ax_val[2][0].set_ylabel("Frequency")
    ax_val[2][0].grid(True)

    ax_val[0][1].hist(speed_val.tolist(), bins=100, range=(-20*1000/3600, 100*1000/3600))
    ax_val[0][1].set_xlabel("speed value")
    ax_val[0][1].set_ylabel("Frequency")
    ax_val[0][1].grid(True)

    ax_val[1][1].scatter(steer_val.tolist(), speed_val.tolist(), s=5)
    ax_val[1][1].set_xlabel("steer")
    ax_val[1][1].set_ylabel("speed")
    ax_val[1][1].grid(True)

    ax_val[2][1].hist(command_val.tolist(), bins=8, range=(-1, 7))
    ax_val[2][1].set_xlabel("command value")
    ax_val[2][1].set_ylabel("Frequency")
    ax_val[2][1].grid(True)

    ax_val[0][2].hist(throttle_val.tolist(), bins=100, range=(-1, 1))
    ax_val[0][2].set_xlabel("throttle")
    ax_val[0][2].set_ylabel("Frequency")
    ax_val[0][2].grid(True)

    ax_val[1][2].hist(brake_val.tolist(), bins=100, range=(-1, 1))
    ax_val[1][2].set_xlabel("brake")
    ax_val[1][2].set_ylabel("Frequency")
    ax_val[1][2].grid(True)

    plt.show()
    plt.pause()

  def plot_loss(self):
    if not hasattr(self, 'ax_loss'):
      fig_loss, self.ax_loss = plt.subplots(1, 1)
    self.ax_loss.clear()
    self.ax_loss.plot(range(self.cur_epoch+1), self.train_loss, label = "train_loss")
    if len(self.dataloader_val) != 0:
      self.ax_loss.plot(range(self.cur_epoch+1), self.val_loss, label = "val_loss")
    self.ax_loss.set_xlabel("epoch")
    self.ax_loss.set_ylabel("loss")
    self.ax_loss.grid(True)
    self.ax_loss.legend()

    plt.show(block=False)
    plt.pause(1)

  def plot_metrics(self):
    if not hasattr(self, 'ax_metrics'):
      num_metrics = len(self.Train_config.plot_metrics)
      rows = min(4, num_metrics)
      cols = math.ceil(num_metrics / rows)

      base_w = 8
      base_h = 2
      figsize = (cols * base_w, rows * base_h)

      _, self.ax_metrics = plt.subplots(rows, cols, figsize=figsize)
      self.ax_metrics = np.array(self.ax_metrics).reshape(-1)

    graph_idx = 0
    for key in self.Train_config.plot_metrics:
      self.ax_metrics[graph_idx].clear()
      metrics = self.train_metrics[key]
      if isinstance(metrics[0], (list, tuple, np.ndarray)):
        cmap = plt.get_cmap("tab10", len(metrics[0])).colors

        if hasattr(self, "val_metrics"):
          val_metrics = list(zip(*self.val_metrics[key]))
        
        for class_idx, metrics_class in enumerate(zip(*metrics)):
          color = cmap[class_idx]
          self.ax_metrics[graph_idx].plot(range(len(metrics)), metrics_class, color=color, linestyle='-', label = f"Class_{class_idx}_Train")
          if hasattr(self, "val_metrics"):
            self.ax_metrics[graph_idx].plot(range(len(metrics)), val_metrics[class_idx], color=color, linestyle='--', label = f"Class_{class_idx}_Val")
      else:
        self.ax_metrics[graph_idx].plot(range(len(metrics)), metrics, label = "Train")
        if hasattr(self, "val_metrics"):
          self.ax_metrics[graph_idx].plot(range(len(metrics)), self.val_metrics[key], label = "Val")
      self.ax_metrics[graph_idx].set_xlabel("epoch")
      self.ax_metrics[graph_idx].set_ylabel(key)
      self.ax_metrics[graph_idx].grid(True)
      self.ax_metrics[graph_idx].legend()
      graph_idx += 1

    plt.show(block=False)
    plt.pause(1)

  def save(self, model_save_dir, config_set, epoch = None):
    if epoch is not None:
      model_file = os.path.join(model_save_dir, f"model_{epoch:04d}.pth")
    else:
      model_file = os.path.join(model_save_dir, f'model_latest.pth')
    torch.save(self.model_wrapper.state_dict(), model_file)

    for conf_name, config in config_set.items():
      json_config = jsonpickle.encode(config)
      with open(os.path.join(model_save_dir, f'{conf_name}.json'), 'w') as f2:
        f2.write(json_config)

    self.save_log(model_save_dir, epoch)

  def save_log(self, model_save_dir, epoch = None):

    has_val_loss = hasattr(self, 'val_loss') and len(self.val_loss) > 0
    has_val_metrics = hasattr(self, 'val_metrics') and len(self.val_metrics) > 0
    
    def build_header():
      row = ["epoch", "train_loss"]
      if  has_val_loss:
        row.append("val_loss")
      for key, value in self.train_metrics.items():
        if isinstance(value[0], (list, tuple, np.ndarray)):
          for class_idx in range(len(value[0])):
            row.append(f"{key}_{class_idx}_train")
            if has_val_metrics:
              row.append(f"{key}_{class_idx}_val")
        else:
          row.append(f"{key}_train")
          if has_val_metrics:
              row.append(f"{key}_val")
      return row

    def build_data(epoch):
      data = [epoch, self.train_loss[epoch]]
      if  has_val_loss:
        data.append(self.val_loss[epoch])

      for key, value in self.train_metrics.items():
        if isinstance(value[0], (list, tuple, np.ndarray)):
          for class_idx in range(len(value[0])):
            data.append(value[epoch][class_idx])
            if has_val_metrics:
              data.append(self.val_metrics[key][epoch][class_idx])
        else:
          data.append(value[epoch])
          if has_val_metrics:
            data.append(self.val_metrics[key][epoch])
      return data

    log_path = os.path.join(model_save_dir, 'log.csv')
    if epoch is None:
      with open(log_path, 'w', newline="") as log:
        writer = csv.writer(log)

        row = build_header()
        writer.writerow(row)

        for epoch in range(len(self.train_loss)):
          data = build_data(epoch)
          writer.writerow(data)
    else:
      log_exists = os.path.exists(log_path)
      with open(log_path, 'a', newline="") as log:
        writer = csv.writer(log)

        if not log_exists:
          row = build_header()
          writer.writerow(row)

        data = build_data(epoch)
        writer.writerow(data)

class Model_wrapper:
  def __init__(self, model, config, device):
    self.device = device
    self.model = model.to(device)
    self.config = config
    self.compute_loss = None

  def load_data_compute_loss(self, data):
    pred = torch.tensor
    label = torch.tensor
    loss = torch.tensor
    loss_individual = {} #Return empty when model output is single
    return pred, label, loss, loss_individual
    
  def parameters(self):
    return self.model.parameters()
  
  def train(self):
    self.model.train()

  def eval(self):
    self.model.eval()

  def state_dict(self):
    return self.model.state_dict()
  
  def plot_model_out():
    return

  def init_metrics(self):
    return

  def cal_metrics_batch(self):
    return

  def cal_metrics_epoch(self):
    metrics = {}
    return metrics