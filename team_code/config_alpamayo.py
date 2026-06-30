"""
Config for the Alpamayo data-collection adapter.

Subclasses :class:`config.GlobalConfig` and only adds/overrides what the
7-camera surround rig needs. Everything else (autopilot, controllers,
kinematic model, ...) is inherited unchanged.

The camera roster mirrors NVIDIA Alpamayo 1.5's surround setup and is ordered
to match the model's ``camera_name_to_index`` mapping (see
``alpamayo/src/alpamayo1_5/load_physical_aiavdataset.py``) so the open-loop
adapter can assign camera indices ``[0..6]`` directly:

    0: cross_left_120   1: front_wide_120   2: cross_right_120
    3: rear_left_70     4: rear_tele_30     5: rear_right_70
    6: front_tele_30
"""

from config import GlobalConfig


class AlpamayoConfig(GlobalConfig):
  """GlobalConfig variant that defines a 7-camera surround rig and no augmentation."""

  def __init__(self):
    super().__init__()

    # The released TF++ camera augmentation collects a second, shifted RGB/seg/depth
    # view. Alpamayo only consumes raw RGB, so we disable it (also keeps the parent's
    # augmentation_translation/rotation at 0 in the saved measurements).
    self.augment = 0

    # All seven cameras share one roof mount. The released config mounts the camera
    # at the driver position [-1.5, 0.0, 2.0]; we raise z by ~0.3 m to sit on the roof.
    cam_x = -1.5
    cam_y = 0.0
    cam_z = 2.3
    cam_width = 1024
    cam_height = 512

    # (id, yaw_deg, fov_deg). Yaws are chosen so the field-of-view cones tile the full
    # surround: front/cross 120 deg cover front + sides, the two rear 70 deg + rear
    # 30 deg telephoto close the back. Tune per your target rig if needed.
    camera_layout = [
        ('cam_cross_left', -90.0, 120.0),
        ('cam_front_wide', 0.0, 120.0),
        ('cam_cross_right', 90.0, 120.0),
        ('cam_rear_left', -145.0, 70.0),
        ('cam_rear_tele', 180.0, 30.0),
        ('cam_rear_right', 145.0, 70.0),
        ('cam_front_tele', 0.0, 30.0),
    ]

    self.cameras = [{
        'id': cam_id,
        'x': cam_x,
        'y': cam_y,
        'z': cam_z,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': yaw,
        'width': cam_width,
        'height': cam_height,
        'fov': fov,
    } for (cam_id, yaw, fov) in camera_layout]

    # Simlingo consumes a single front camera with a *different* mount/optics than the
    # Alpamayo surround rig: it sits at the driver height z=2.0 (not the 2.3 roof mount)
    # and uses a 110 deg FOV (not 120). We collect it alongside the rig so a single run
    # feeds both models. The simlingo-open-loop adapter reads cam_front_simlingo/ by name;
    # the Alpamayo adapter selects its four cameras by name and ignores this extra folder.
    self.cameras.append({
        'id': 'cam_front_simlingo',
        'x': -1.5,
        'y': 0.0,
        'z': 2.0,
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0,
        'width': 1024,
        'height': 512,
        'fov': 110.0,
    })
