"""
Alpamayo data-collection adapter.

A trimmed sibling of ``data_agent.DataAgent`` that collects a 7-camera surround
RGB rig matching NVIDIA Alpamayo 1.5, plus the parent autopilot's measurements
and (lidar-free) bounding boxes. LiDAR and BEV semantics are removed; depth is
left in place but commented out so it can be re-enabled later.

Bounding-box, geometry and weather helpers are inherited from ``DataAgent``.
Only the data-collection surface (setup / _init / sensors / tick / run_step /
save_sensors) is overridden.
"""

import os
import gzip
import json
from pathlib import Path

import cv2
import torch

from autopilot import AutoPilot
from data_agent import DataAgent
from config_alpamayo import AlpamayoConfig

from agents.navigation.local_planner import LocalPlanner


def get_entry_point():
  return 'DataAgentAlpamayo'


class DataAgentAlpamayo(DataAgent):
  """Collects a 7-camera surround RGB rig for Alpamayo, without lidar/bev/depth."""

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    # Skip DataAgent.setup (it creates lidar/semantics/depth/bev dirs and expects
    # those sensors); go straight to the base AutoPilot setup, then swap in our config.
    super(DataAgent, self).setup(path_to_conf_file, route_index, traffic_manager=None)
    self.config = AlpamayoConfig()

    # Re-create the handful of instance attributes DataAgent.setup normally sets and
    # that the inherited helpers rely on.
    self.weather_tmp = None
    self.step_tmp = 0
    self.tm = traffic_manager
    self.scenario_name = Path(path_to_conf_file).parent.name
    self.cutin_vehicle_starting_position = None
    self.tmp_visu = int(os.environ.get('TMP_VISU', 0))
    self._active_traffic_light = None

    if self.save_path is not None and self.datagen:
      for camera in self.config.cameras:
        (self.save_path / camera['id']).mkdir()
      (self.save_path / 'boxes').mkdir()

  def _init(self, hd_map):
    # Base init only; skip DataAgent._init (BEV ObsManagers / augmented dummy vehicle).
    super(DataAgent, self)._init(hd_map)
    # NOTE: intentionally do NOT call shuffle_weather() here. DataAgent.shuffle_weather
    # picks a random CARLA weather preset (about half of which are *Night presets) and
    # overrides the simulation weather, ignoring the route XML. We want the <weathers>
    # block in the route file to be honored, so we leave the leaderboard-applied route
    # weather untouched.
    # LocalPlanner is required by the inherited _vehicle_obstacle_detected.
    self._local_planner = LocalPlanner(self._vehicle, opt_dict={}, map_inst=self.world_map)

  def sensors(self):
    result = super(DataAgent, self).sensors()  # hd_map, imu, speedometer

    if self.save_path is not None and (self.datagen or self.tmp_visu):
      for camera in self.config.cameras:
        result.append({
            'type': 'sensor.camera.rgb',
            'x': camera['x'],
            'y': camera['y'],
            'z': camera['z'],
            'roll': camera['roll'],
            'pitch': camera['pitch'],
            'yaw': camera['yaw'],
            'width': camera['width'],
            'height': camera['height'],
            'fov': camera['fov'],
            'id': camera['id'],
        })

      # --- Depth cameras (disabled). Uncomment together with the depth blocks in
      # --- tick() and save_sensors() to also collect aligned per-camera depth.
      # for camera in self.config.cameras:
      #   result.append({
      #       'type': 'sensor.camera.depth',
      #       'x': camera['x'],
      #       'y': camera['y'],
      #       'z': camera['z'],
      #       'roll': camera['roll'],
      #       'pitch': camera['pitch'],
      #       'yaw': camera['yaw'],
      #       'width': camera['width'],
      #       'height': camera['height'],
      #       'fov': camera['fov'],
      #       'id': f"{camera['id']}_depth",
      #   })

    return result

  def tick(self, input_data):
    result = {}

    if self.save_path is not None and (self.datagen or self.tmp_visu):
      rgb = {camera['id']: input_data[camera['id']][1][:, :, :3] for camera in self.config.cameras}

      # --- Depth (disabled). See sensors() / save_sensors().
      # depth = {}
      # for camera in self.config.cameras:
      #   raw = input_data[f"{camera['id']}_depth"][1][:, :, :3]
      #   depth[camera['id']] = (t_u.convert_depth(raw) * 255.0 + 0.5).astype(np.uint8)
    else:
      rgb = None

    # No lidar: bounding boxes are saved without per-box lidar hit counts (num_points = -1).
    bounding_boxes = self.get_bounding_boxes(lidar=None)

    result.update({
        'rgb': rgb,
        'bounding_boxes': bounding_boxes,
    })

    return result

  @torch.inference_mode()
  def run_step(self, input_data, timestamp, sensors=None, plant=False):
    self.step_tmp += 1

    # No lidar conversion and no camera augmentation (augment disabled in config).
    control = super(DataAgent, self).run_step(input_data, timestamp, plant=plant)

    tick_data = self.tick(input_data)

    if self.step % self.config.data_save_freq == 0:
      if self.save_path is not None and self.datagen:
        self.save_sensors(tick_data)

    if plant:
      # Control contains data when run with plant
      return {**tick_data, **control}
    else:
      return control

  def save_sensors(self, tick_data):
    frame = self.step // self.config.data_save_freq

    # CARLA images are already in opencv's BGR format.
    for cam_id, image in tick_data['rgb'].items():
      cv2.imwrite(str(self.save_path / cam_id / (f'{frame:04}.jpg')), image)

    # --- Depth saving (disabled). See sensors() / tick().
    # for cam_id, depth in tick_data['depth'].items():
    #   cv2.imwrite(str(self.save_path / f'{cam_id}_depth' / (f'{frame:04}.png')), depth)

    with gzip.open(self.save_path / 'boxes' / (f'{frame:04}.json.gz'), 'wt', encoding='utf-8') as f:
      json.dump(tick_data['bounding_boxes'], f, indent=4)
