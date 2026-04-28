import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from magnum import Vector3

import habitat_sim as hs

with open(
    "/developer/ros2_hydra_ws/src/habitat-ros/config/hm3dall_label_space.yaml", "r"
) as f:
    labelspace = yaml.safe_load(f)
labelspace_map = {
    label_name["name"]: label_name["label"] for label_name in labelspace["label_names"]
}
inverse_labelspace_map = {v: k for k, v in labelspace_map.items()}

# Read csv colormap with entries: name,red,green,blue,aplha,id. And create a mapping from id to rgb color.
colormap = {}
with open(
    "/developer/ros2_hydra_ws/src/habitat-ros/config/distinct_all_hm3d_colors.csv", "r"
) as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        name, red, green, blue, alpha, id = line.strip().split(",")
        colormap[int(id)] = (int(red), int(green), int(blue), int(alpha))


def f_to_hfov(f: float, width: int) -> float:
    """Convert focal length in pixels to horizontal field of view in degrees.
    https://github.com/facebookresearch/habitat-sim/issues/402"""
    return math.degrees(2.0 * math.atan(float(width) / (2.0 * f)))


def rgb_sensor_config() -> hs.CameraSensorSpec:
    """Return the configuration for a Habitat color sensor."""
    rgb_sensor_spec = hs.CameraSensorSpec()
    rgb_sensor_spec.uuid = str(hs.SensorType.COLOR)
    rgb_sensor_spec.sensor_type = hs.SensorType.COLOR
    rgb_sensor_spec.sensor_subtype = hs.SensorSubType.PINHOLE
    rgb_sensor_spec.resolution = [480, 640]
    rgb_sensor_spec.near = 0.00001
    rgb_sensor_spec.far = 1000
    rgb_sensor_spec.hfov = f_to_hfov(525.0, 640)
    rgb_sensor_spec.position = Vector3(0.0, 0.0, 0.0)
    rgb_sensor_spec.orientation = Vector3(0.0, 0.0, 0.0)
    return rgb_sensor_spec


def depth_sensor_config() -> hs.CameraSensorSpec:
    """Return the configuration for a Habitat depth sensor."""
    depth_sensor_spec = hs.CameraSensorSpec()
    depth_sensor_spec.uuid = str(hs.SensorType.DEPTH)
    depth_sensor_spec.sensor_type = hs.SensorType.DEPTH
    depth_sensor_spec.sensor_subtype = hs.SensorSubType.PINHOLE
    depth_sensor_spec.resolution = [480, 640]
    depth_sensor_spec.near = 0.1
    depth_sensor_spec.far = 20.0
    depth_sensor_spec.hfov = f_to_hfov(525.0, 640)
    depth_sensor_spec.position = Vector3(0.0, 0.0, 0.0)
    depth_sensor_spec.orientation = Vector3(0.0, 0.0, 0.0)
    if False:
        depth_sensor_spec.noise_model = "RedwoodDepthNoiseModel"
    return depth_sensor_spec


def semantic_sensor_config() -> hs.CameraSensorSpec:
    """Return the configuration for a Habitat semantic sensor."""
    semantic_sensor_spec = hs.CameraSensorSpec()
    semantic_sensor_spec.uuid = str(hs.SensorType.SEMANTIC)
    semantic_sensor_spec.sensor_type = hs.SensorType.SEMANTIC
    semantic_sensor_spec.sensor_subtype = hs.SensorSubType.PINHOLE
    semantic_sensor_spec.resolution = [480, 640]
    semantic_sensor_spec.near = 0.00001
    semantic_sensor_spec.far = 1000
    semantic_sensor_spec.hfov = f_to_hfov(525.0, 640)
    semantic_sensor_spec.position = Vector3(0.0, 0.0, 0.0)
    semantic_sensor_spec.orientation = Vector3(0.0, 0.0, 0.0)
    return semantic_sensor_spec


backend_config = hs.SimulatorConfiguration()
backend_config.scene_id = "/developer/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
backend_config.enable_physics = False
backend_config.allow_sliding = False
backend_config.scene_dataset_config_file = (
    "/developer/hm3d/val/hm3d_annotated_val_basis.scene_dataset_config.json"
)
agent_config = hs.AgentConfiguration()
backend_config.gpu_device_id = -1  # Use CPU
agent_config.sensor_specifications = [
    rgb_sensor_config(),
    depth_sensor_config(),
    semantic_sensor_config(),
]

agent_config.height = 0.0
agent_config.radius = 0.0
sim = hs.Simulator(hs.Configuration(backend_config, [agent_config]))

object_to_cat_map = {c.id: c.category.index() for c in sim.semantic_scene.objects}
category_map = np.array(list(object_to_cat_map.values()))
name_mapping = {}
scene_label_2_cat_map = {}
for c in sim.semantic_scene.categories:
    name_mapping[c.index()] = c.name()
    scene_label_2_cat_map[c.index()] = labelspace_map[c.name().lower()]

hm3d_cat_idxs = sorted(list(name_mapping.keys()))
names = [name_mapping[idx] for idx in hm3d_cat_idxs]

agent = sim.get_agent(0)
observations = sim.get_sensor_observations()
depth = observations[str(hs.SensorType.DEPTH)]
rgb = observations[str(hs.SensorType.COLOR)][:, :, :3]
instance_labels = observations[str(hs.SensorType.SEMANTIC)]
# Apply category mapping to instance labels
labels = np.asarray(
    [scene_label_2_cat_map[label] for label in category_map[instance_labels].flatten()]
).reshape(instance_labels.shape)
# Apply colormap to instance labels
color_labels = np.array([colormap[label] for label in labels.flatten()]).reshape(
    instance_labels.shape[0], instance_labels.shape[1], 4
)


def add_centered_semantic_labels(
    semantic_img: np.ndarray,
    semantic_labels: np.ndarray,
    label_name_map: dict[int, str],
) -> np.ndarray:
    """Overlay label names at the centroid of each visible semantic class."""
    annotated = semantic_img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.3
    thickness = 1

    for label_id in np.unique(semantic_labels):
        label_name = label_name_map.get(int(label_id))
        if label_name is None:
            continue

        ys, xs = np.where(semantic_labels == label_id)
        if len(xs) == 0:
            continue

        center_x = int(xs.mean())
        center_y = int(ys.mean())
        (text_w, text_h), baseline = cv2.getTextSize(
            label_name, font, font_scale, thickness
        )
        origin_x = max(0, min(center_x - text_w // 2, annotated.shape[1] - text_w))
        origin_y = max(
            text_h, min(center_y + text_h // 2, annotated.shape[0] - baseline)
        )

        cv2.putText(
            annotated,
            label_name,
            (origin_x, origin_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            label_name,
            (origin_x, origin_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated


annotated_color_labels = add_centered_semantic_labels(
    color_labels[:, :, :3].astype(np.uint8),
    labels,
    inverse_labelspace_map,
)

fig, ax = plt.subplots(1, 4, figsize=(15, 5))
ax[0].imshow(rgb)
ax[0].set_title("RGB")
ax[1].imshow(depth, cmap="gray")
ax[1].set_title("Depth")
ax[2].imshow(annotated_color_labels)
ax[2].set_title("Semantic")
ax[3].imshow(instance_labels)
ax[3].set_title("Instance")
fig.savefig("/developer/ros2_hydra_ws/img.png")
