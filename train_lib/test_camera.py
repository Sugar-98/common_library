from common.config import GlobalConfig
import numpy as np
from common.utils import conv_NuScenes2Carla

config = GlobalConfig()

result = []

for idx, camera in enumerate(config.cameras):
          tmp_trans, tmp_rot = conv_NuScenes2Carla(camera.trans, camera.get_rot_mat())
          result.append({
          'type': 'sensor.camera.rgb',
          'x': tmp_trans[0],
          'y': tmp_trans[1],
          'z': tmp_trans[2],
          'roll': np.degrees(tmp_rot[0]),
          'pitch': np.degrees(tmp_rot[1]),
          'yaw': np.degrees(tmp_rot[2]),
          'width': camera.W,
          'height': camera.H,
          'fov': np.degrees(camera.fov),
          'id': camera.name
          })

print(result)