#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020-2021 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020-2021 Sotiris Papatheodorou
# BSD 3-Clause License

# Copyright (c) 2025, NTNU Autonomous Robots Lab
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Habitat-Sim ROS2 node."""

import math
import os
import pathlib
import threading
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import quaternion
import rclpy
import tf2_ros
import yaml
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped, Transform, TransformStamped
from magnum import Vector3
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image

import habitat_ros
import habitat_sim as hs
from habitat_ros import LoggerInfo, LoggerWarn

# Custom type definitions
Config = Dict[str, Any]
Observation = hs.sensor.Observation
Publishers = Dict[str, Any]
Sim = hs.Simulator


def split_pose(T: np.array) -> Tuple[np.array, quaternion.quaternion]:
    """Split a pose in a 4x4 matrix into a position vector and an orientation
    quaternion."""
    return T[0:3, 3], quaternion.from_rotation_matrix(T[0:3, 0:3]).normalized()


def print_config(config: Config) -> None:
    """Print a dictionary containing the configuration to the ROS info log."""
    for name, val in config.items():
        LoggerInfo.info("  {: <20} {}".format(name + ":", str(val)))


def combine_pose(t: np.array, q: quaternion.quaternion) -> np.array:
    """Combine a position vector and an orientation quaternion into a 4x4 pose
    matrix."""
    T = np.identity(4)
    T[0:3, 3] = t
    T[0:3, 0:3] = quaternion.as_rotation_matrix(q.normalized())
    return T


def msg_to_pose(msg: Pose) -> np.array:
    """Convert a ROS Pose message to a 4x4 pose matrix."""
    t = [msg.position.x, msg.position.y, msg.position.z]
    q = quaternion.quaternion(
        msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z
    ).normalized()
    return combine_pose(t, q)


def msg_to_transform(msg: Transform) -> np.array:
    """Convert a ROS Transform message to a 4x4 transform matrix."""
    t = [msg.translation.x, msg.translation.y, msg.translation.z]
    q = quaternion.quaternion(
        msg.rotation.w, msg.rotation.x, msg.rotation.y, msg.rotation.z
    ).normalized()
    return combine_pose(t, q)


def hfov_to_f(hfov: float, width: int) -> float:
    """Convert horizontal field of view in degrees to focal length in pixels.
    https://github.com/facebookresearch/habitat-sim/issues/402"""
    return 1.0 / (2.0 / float(width) * math.tan(math.radians(hfov) / 2.0))


def f_to_hfov(f: float, width: int) -> float:
    """Convert focal length in pixels to horizontal field of view in degrees.
    https://github.com/facebookresearch/habitat-sim/issues/402"""
    return math.degrees(2.0 * math.atan(float(width) / (2.0 * f)))


def find_tf(
    tf_buffer: tf2_ros.Buffer, from_frame: str, to_frame: str
) -> Union[np.array, None]:
    """Return the transformation relating the 2 frames (ROS2 version)."""
    try:
        # timeout of 0.01 seconds
        tf_msg = tf_buffer.lookup_transform(
            from_frame,
            to_frame,
            Time(seconds=0),  # use latest
            timeout=Duration(seconds=0.01),  # small timeout
        )
        return msg_to_transform(tf_msg.transform)
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ) as e:
        print(
            f'FATAL: Could not find transform from frame "{from_frame}" to frame "{to_frame}"'
        )
        raise e


def list_to_pose(lst: List) -> Union[np.array, None]:
    """Convert a list to a pose represented by a 4x4 homogeneous matrix. The
    list may have a varying number of elements:
    - 3 (translation: x, y, z)
    - 4 (orientation quaternion: qx, qy, qz, qw)
    - 7 (translation, orientation quaternion)
    - 16 (4x4 homogeneous matrix in row-major from)"""
    n = len(lst)
    if n == 3:
        # Position: tx, ty, tz
        T = np.identity(4)
        T[0:3, 3] = np.array(lst).T
    elif n == 4:
        # Orientation quaternion: qx, qy, qz, qw
        q = quaternion.quaternion(lst[3], lst[0], lst[1], lst[2]).normalized()
        T = np.identity(4)
        T[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    elif n == 7:
        # Position and orientation quaternion: tx, ty, tz, qx, qy, qz, qw
        q = quaternion.quaternion(lst[6], lst[3], lst[4], lst[5]).normalized()
        T = np.identity(4)
        T[0:3, 3] = np.array(lst[0:3]).T
        T[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    elif n == 16:
        # 4x4 pose matrix in row-major order
        T = np.array(lst)
        T = T.reshape((4, 4))
        LoggerWarn.warning(T)
    else:
        T = None
    return T


def remove_invalid_objects(
    objects: List[hs.scene.SemanticObject],
) -> List[hs.scene.SemanticObject]:
    return [x for x in objects if x is not None and x.category is not None]


def get_instance_id(o: hs.scene.SemanticObject) -> int:
    s = o.id.strip("_")
    if "_" in s:
        return [int(x) for x in s.split("_")][2]
    else:
        return int(s)


class HabitatROSNode(Node):
    # Matterport3D class RGB colors
    class_colors = np.array(
        [
            [0xFF, 0xFF, 0xFF],
            [0xAE, 0xC7, 0xE8],
            [0x70, 0x80, 0x90],
            [0x98, 0xDF, 0x8A],
            [0xC5, 0xB0, 0xD5],
            [0xFF, 0x7F, 0x0E],
            [0xD6, 0x27, 0x28],
            [0x1F, 0x77, 0xB4],
            [0xBC, 0xBD, 0x22],
            [0xFF, 0x98, 0x96],
            [0x2C, 0xA0, 0x2C],
            [0xE3, 0x77, 0xC2],
            [0xDE, 0x9E, 0xD6],
            [0x94, 0x67, 0xBD],
            [0x8C, 0xA2, 0x52],
            [0x84, 0x3C, 0x39],
            [0x9E, 0xDA, 0xE5],
            [0x9C, 0x9E, 0xDE],
            [0xE7, 0x96, 0x9C],
            [0x63, 0x79, 0x39],
            [0x8C, 0x56, 0x4B],
            [0xDB, 0xDB, 0x8D],
            [0xD6, 0x61, 0x6B],
            [0xCE, 0xDB, 0x9C],
            [0xE7, 0xBA, 0x52],
            [0x39, 0x3B, 0x79],
            [0xA5, 0x51, 0x94],
            [0xAD, 0x49, 0x4A],
            [0xB5, 0xCF, 0x6B],
            [0x52, 0x54, 0xA3],
            [0xBD, 0x9E, 0x39],
            [0xC4, 0x9C, 0x94],
            [0xF7, 0xB6, 0xD2],
            [0x6B, 0x6E, 0xCF],
            [0xFF, 0xBB, 0x78],
            [0xC7, 0xC7, 0xC7],
            [0x8C, 0x6D, 0x31],
            [0xE7, 0xCB, 0x94],
            [0xCE, 0x6D, 0xBD],
            [0x17, 0xBE, 0xCF],
            [0x7F, 0x7F, 0x7F],
        ]
    )

    # Instantiate a single CvBridge object for all conversions
    _bridge = CvBridge()
    # Published topic names
    _rgb_topic_name = "rgb/"
    _depth_topic_name = "depth/"
    _sem_class_topic_name = "semantic_class/"
    _sem_instance_topic_name = "semantic_instance/"
    _habitat_pose_topic_name = "pose"
    # Subscribed topic names
    _external_pose_topic_name = "external_pose"

    # Transforms between the internal habitat frame I (y-up) and the exported
    # habitat frame H (z-up)
    _T_HI = np.identity(4)
    _T_HI[0:3, 0:3] = quaternion.as_rotation_matrix(
        hs.utils.common.quat_from_two_vectors(
            np.array([hs.geo.GRAVITY.x, hs.geo.GRAVITY.y, hs.geo.GRAVITY.z]),
            np.array([0.0, 0.0, -1.0]),
        )
    )
    _T_IH = np.linalg.inv(_T_HI)

    # Transforms between the habitat camera frame C (-z-forward, y-up) and the
    # ROS body frame B (x-forward, z-up)
    _T_CB = np.array(
        [
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ]
    )
    _T_BC = np.linalg.inv(_T_CB)

    # Transforms between the TUM camera frame Ctum (z-forward, x-right) and the
    # ROS body frame B (x-forward, z-up)
    _T_BCtum = np.array(
        [
            (0.0, 0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ]
    )

    _T_RC = np.array(
        [
            (0.0, 0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ]
    )

    # The default node options
    _default_config = {
        "width": 640,
        "height": 480,
        "near_plane": 0.1,
        "far_plane": 10.0,
        "f": 525.0,
        "fps": 30,
        "enable_semantics": False,
        "depth_noise": False,
        "allowed_classes": [],
        "scene_file": "",
        "initial_T_HB": [],
        "pose_frame_id": "habitat",
        "pose_frame_at_initial_T_HB": False,
        "visualize_semantics": False,
        "recording_dir": "",
        "world_frame": "world",
        "robot_frame": "base_link",
        "sensor_frame": "camera_link",
    }

    def __init__(self):
        super().__init__("habitat_node")

        # Declare parameters
        config_path = (
            self.declare_parameter("config_path", "").get_parameter_value().string_value
        )
        config_path = pathlib.Path(config_path).expanduser().absolute()
        # Read config
        self.config = self._read_node_config(config_path)

        # Init Habitat simulator
        self.sim = self._init_habitat(self.config)

        # Publishers
        self.pub = self._init_publishers(self.config)

        # Mutex for pose
        self.T_HB_mutex = threading.Lock()

        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Static TF: robot_frame -> camera_link
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # Robot to camera static transform
        msg = self._transform_to_msg(
            self._T_RC,
            self.config["robot_frame"],  # parent
            self.config["sensor_frame"],  # child
        )
        self.tf_static_broadcaster.sendTransform(msg)

        # Habitat to world static transform
        if self.config["world_frame"] != "habitat":
            msg = self._transform_to_msg(
                np.eye(4),
                "habitat",
                self.config["world_frame"],
            )
            self.tf_static_broadcaster.sendTransform(msg)

        # Pose frame static broadcaster if needed
        if (
            self.config["pose_frame_at_initial_T_HB"]
            and self.config["pose_frame_id"] != "habitat"
        ):
            self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
            T_HP_msg = self._transform_to_msg(
                self.T_HB, "habitat", self.config["pose_frame_id"]
            )
            self.tf_static_broadcaster.sendTransform(T_HP_msg)

        # Subscribe to external pose
        self.create_subscription(
            PoseStamped, self._external_pose_topic_name, self._pose_callback, 1
        )

        self.get_logger().info("Habitat node ready")

        # Timer loop
        if self.config["fps"] > 0:
            period = 1.0 / self.config["fps"]
            self.timer = self.create_timer(period, self._main_loop)

    def _main_loop(self) -> None:
        """Main loop: move the agent, render and publish the observation, and
        record if needed."""
        observation = self._move_and_render(self.sim, self.config)
        self._publish_observation(observation, self.pub, self.config)
        if self.config["recording_dir"]:
            self._record_observation(observation, self.config["recording_dir"])

    def _read_node_config(self, config_path: pathlib.Path) -> Config:
        """Read the node parameters, print them and return a dictionary."""
        config = self._default_config.copy()

        # Read the parameters

        if config_path.exists():
            with config_path.open("r") as f:
                file_config = yaml.safe_load(f)
            config.update(file_config)
            self.get_logger().info(f"Loaded config from '{config_path}'")
        else:
            self.get_logger().warn(f"Config path '{config_path}' does not exist!")

        # Get an absolute path from the supplied scene file
        config["scene_file"] = os.path.expanduser(config["scene_file"])
        if not os.path.isabs(config["scene_file"]):
            # The scene file path is relative, assuming relative to the ROS package
            package_path = get_package_share_directory("habitat_ros") + "/"
            config["scene_file"] = package_path + config["scene_file"]

        # Ensure a valid scene file was supplied
        if not config["scene_file"] or not os.path.isfile(config["scene_file"]):
            raise RuntimeError("Scene file missing or invalid: " + config["scene_file"])

        self.get_logger().info("Habitat node parameters:")
        for name, val in config.items():
            self.get_logger().info(f"  {name}: {val}")

        # Create the initial T_HB matrix
        T = list_to_pose(config["initial_T_HB"])
        if T is None and config["initial_T_HB"]:
            self.get_logger().error(
                "Invalid initial T_HB. Expected list of 3, 4, 7 or 16 elements"
            )
        config["initial_T_HB"] = T
        if config["recording_dir"]:
            config["recording_dir"] = os.path.expanduser(config["recording_dir"])
        self.get_logger().info("Habitat node parameters:")
        print_config(config)
        return config

    def _broadcast_tf(self, T_HB: np.array) -> None:
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.config["world_frame"]  # e.g. "habitat"
        msg.child_frame_id = self.config["robot_frame"]  # "base_link"

        msg.transform.translation.x = float(T_HB[0, 3])
        msg.transform.translation.y = float(T_HB[1, 3])
        msg.transform.translation.z = float(T_HB[2, 3])

        q = quaternion.from_rotation_matrix(T_HB[0:3, 0:3]).normalized()
        msg.transform.rotation.x = float(q.x)
        msg.transform.rotation.y = float(q.y)
        msg.transform.rotation.z = float(q.z)
        msg.transform.rotation.w = float(q.w)

        self.tf_broadcaster.sendTransform(msg)

    def _transform_to_msg(
        self, T_TF: np.array, from_frame: str, to_frame: str
    ) -> TransformStamped:
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = from_frame
        msg.child_frame_id = to_frame
        msg.transform.translation.x = T_TF[0, 3]
        msg.transform.translation.y = T_TF[1, 3]
        msg.transform.translation.z = T_TF[2, 3]
        q_TF = quaternion.from_rotation_matrix(T_TF[0:3, 0:3]).normalized()
        msg.transform.rotation.x = q_TF.x
        msg.transform.rotation.y = q_TF.y
        msg.transform.rotation.z = q_TF.z
        msg.transform.rotation.w = q_TF.w
        return msg

    def _init_habitat(self, config: Config) -> Sim:
        """Initialize the Habitat simulator, create the sensors and load the
        scene file."""
        backend_config = hs.SimulatorConfiguration()
        backend_config.scene_id = config["scene_file"]
        agent_config = hs.AgentConfiguration()
        backend_config.gpu_device_id = -1  # Use CPU
        agent_config.sensor_specifications = [
            self._rgb_sensor_config(config),
            self._depth_sensor_config(config),
            self._semantic_sensor_config(config),
        ]
        agent_config.height = 0.0
        agent_config.radius = 0.0
        sim = Sim(hs.Configuration(backend_config, [agent_config]))
        # Get the intrinsic camera parameters
        hfov = float(agent_config.sensor_specifications[0].hfov)
        f = hfov_to_f(hfov, config["width"])
        cx = config["width"] / 2.0 - 0.5
        cy = config["height"] / 2.0 - 0.5
        config["K"] = np.array(
            [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        config["P"] = np.array(
            [[f, 0.0, cx, 0.0], [0.0, f, cy, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        self.class_id_to_name = self._class_id_to_name_map(
            sim.semantic_scene.categories
        )
        # Setup the instance/class conversion map
        if config["enable_semantics"]:
            config["instance_to_class"] = self._instance_to_class_map(
                remove_invalid_objects(sim.semantic_scene.objects),
                self.class_id_to_name,
            )
            if config["instance_to_class"].size == 0:
                self.get_logger().warn("The scene contains no semantics")
        # Get or set the initial agent pose
        agent = sim.get_agent(0)
        if config["initial_T_HB"] is None:
            t_IC = agent.get_state().position
            q_IC = agent.get_state().rotation
            T_IC = combine_pose(t_IC, q_IC)
            self.T_HB = self._T_IC_to_T_HB(T_IC)
        else:
            self.T_HB = config["initial_T_HB"]
            t_IC, q_IC = split_pose(self._T_HB_to_T_IC(self.T_HB))
            agent_state = hs.agent.AgentState(t_IC, q_IC)
            agent.set_state(agent_state)
        t_HB, q_HB = split_pose(self.T_HB)
        # Initialize the current pose timestamp to zero.
        self.T_HB_stamp = Time(nanoseconds=0)
        self.T_HB_received = False
        self.get_logger().info(
            "Habitat initial t_HB (x,y,z):   {}, {}, {}".format(
                t_HB[0], t_HB[1], t_HB[2]
            )
        )
        self.get_logger().info(
            "Habitat initial q_HB (x,y,z,w): {}, {}, {}, {}".format(
                q_HB.x, q_HB.y, q_HB.z, q_HB.w
            )
        )
        return sim

    def _rgb_sensor_config(self, config: Config) -> hs.CameraSensorSpec:
        """Return the configuration for a Habitat color sensor."""
        rgb_sensor_spec = hs.CameraSensorSpec()
        rgb_sensor_spec.uuid = "rgb"
        rgb_sensor_spec.sensor_type = hs.SensorType.COLOR
        rgb_sensor_spec.sensor_subtype = hs.SensorSubType.PINHOLE
        rgb_sensor_spec.resolution = [config["height"], config["width"]]
        rgb_sensor_spec.near = 0.00001
        rgb_sensor_spec.far = 1000
        rgb_sensor_spec.hfov = f_to_hfov(config["f"], config["width"])
        rgb_sensor_spec.position = Vector3(0.0, 0.0, 0.0)
        rgb_sensor_spec.orientation = Vector3(0.0, 0.0, 0.0)
        return rgb_sensor_spec

    def _depth_sensor_config(self, config: Config) -> hs.CameraSensorSpec:
        """Return the configuration for a Habitat depth sensor."""
        depth_sensor_spec = hs.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth"
        depth_sensor_spec.sensor_type = hs.SensorType.DEPTH
        depth_sensor_spec.sensor_subtype = hs.SensorSubType.PINHOLE
        depth_sensor_spec.resolution = [config["height"], config["width"]]
        depth_sensor_spec.near = config["near_plane"]
        depth_sensor_spec.far = config["far_plane"]
        depth_sensor_spec.hfov = f_to_hfov(config["f"], config["width"])
        depth_sensor_spec.position = Vector3(0.0, 0.0, 0.0)
        depth_sensor_spec.orientation = Vector3(0.0, 0.0, 0.0)
        if config["depth_noise"]:
            depth_sensor_spec.noise_model = "RedwoodDepthNoiseModel"
        return depth_sensor_spec

    def _semantic_sensor_config(self, config: Config) -> hs.CameraSensorSpec:
        """Return the configuration for a Habitat semantic sensor."""
        semantic_sensor_spec = hs.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic"
        semantic_sensor_spec.sensor_type = hs.SensorType.SEMANTIC
        semantic_sensor_spec.sensor_subtype = hs.SensorSubType.PINHOLE
        semantic_sensor_spec.resolution = [config["height"], config["width"]]
        semantic_sensor_spec.near = 0.00001
        semantic_sensor_spec.far = 1000
        semantic_sensor_spec.hfov = f_to_hfov(config["f"], config["width"])
        semantic_sensor_spec.position = Vector3(0.0, 0.0, 0.0)
        semantic_sensor_spec.orientation = Vector3(0.0, 0.0, 0.0)
        return semantic_sensor_spec

    def _class_id_to_name_map(self, categories: List) -> Dict[int, str]:
        """Generate a dictionary from class IDs to class names."""
        return {x.index(): x.name() for x in categories if x is not None}

    def _instance_to_class_map(
        self, objects: List[hs.scene.SemanticObject], classes: Dict[int, str]
    ) -> np.ndarray:
        """Given the objects in the scene, create an array that maps instance
        IDs to class IDs."""
        # Default is -1 so that an empty array is created in the following line
        # if there are no objects.
        max_instance_id = max([get_instance_id(x) for x in objects], default=-1)
        mapping = np.zeros(max_instance_id + 1, dtype=np.uint8)
        for object in objects:
            instance_id = get_instance_id(object)
            mapping[instance_id] = object.category.index()
            if mapping[instance_id] not in classes.keys():
                self.get_logger().warn(
                    'Invalid object class ID/name {}/"{}", replacing with 0/"{}"'.format(
                        mapping[instance_id], object.category.name(), classes[0]
                    )
                )
                mapping[instance_id] = 0
        return mapping

    def _init_publishers(self, config: Config) -> Publishers:
        """Initialize and return the image and pose publishers."""
        image_queue_size = 10
        pub = {}
        # Pose publisher
        pub["pose"] = self.create_publisher(
            PoseStamped, self._habitat_pose_topic_name, 10
        )
        # Image publishers
        pub["rgb"] = self.create_publisher(
            Image, self._rgb_topic_name + "image_raw", image_queue_size
        )
        pub["depth"] = self.create_publisher(
            Image, self._depth_topic_name + "image_raw", image_queue_size
        )
        if config["enable_semantics"] and config["instance_to_class"].size > 0:
            # Only publish semantics if the scene contains semantics
            pub["sem_class"] = self.create_publisher(
                Image, self._sem_class_topic_name + "image_raw", image_queue_size
            )
            pub["sem_instance"] = self.create_publisher(
                Image, self._sem_instance_topic_name + "image_raw", image_queue_size
            )
            if config["visualize_semantics"]:
                pub["sem_class_render"] = self.create_publisher(
                    Image,
                    self._sem_class_topic_name + "image_color",
                    image_queue_size,
                )
                pub["sem_instance_render"] = self.create_publisher(
                    Image,
                    self._sem_instance_topic_name + "image_color",
                    image_queue_size,
                )
        # Publish the camera info for each image topic
        image_topics = [self._rgb_topic_name, self._depth_topic_name]
        if config["enable_semantics"] and config["instance_to_class"].size > 0:
            image_topics += [self._sem_class_topic_name, self._sem_instance_topic_name]
        for topic in image_topics:
            qos = QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            pub[topic + "_camera_info"] = self.create_publisher(
                CameraInfo, topic + "camera_info", qos
            )
            pub[topic + "_camera_info"].publish(self._camera_intrinsics_to_msg(config))

        return pub

    def _pose_callback(self, pose: PoseStamped) -> None:
        """Callback for receiving an external pose."""
        """Callback for receiving external pose messages. It updates the agent
        pose."""
        # Find the transform from the pose frame F to the habitat frame H
        T_HE = find_tf(self.tf_buffer, "habitat", pose.header.frame_id)
        # Transform the pose
        T_EB = msg_to_pose(pose.pose)
        T_HB = T_HE @ T_EB
        # Update the pose
        self.T_HB_mutex.acquire()
        self.T_HB = T_HB
        self.T_HB_stamp = pose.header.stamp
        self.T_HB_received = True
        self.T_HB_mutex.release()

    def _filter_sem_classes(self, observation: Observation) -> None:
        """Remove object detections whose classes are not in the allowed class
        list. Their class and instance IDs are set to 0."""
        # Generate a per-pixel boolean matrix
        allowed = np.vectorize(lambda x: x in self.config["allowed_classes"])
        allowed_pixels = allowed(observation["sem_classes"])
        # Set all False pixels to 0 on the class and instance images
        class_zeros = np.zeros(
            observation["sem_classes"].shape, dtype=observation["sem_classes"].dtype
        )
        instance_zeros = np.zeros(
            observation["sem_instances"].shape, dtype=observation["sem_instances"].dtype
        )
        observation["sem_classes"] = np.where(
            allowed_pixels, observation["sem_classes"], class_zeros
        )
        observation["sem_instances"] = np.where(
            allowed_pixels, observation["sem_instances"], instance_zeros
        )

    def _pose_to_msg(self, observation: Observation) -> PoseStamped:
        """Convert the agent pose from the observation to a ROS PoseStamped
        message."""
        T_PH = find_tf(self.tf_buffer, self.config["pose_frame_id"], "habitat")
        t_PB, q_PB = split_pose(T_PH @ observation["T_HB"])
        p = PoseStamped()
        p.header.stamp = observation["timestamp"]
        p.header.frame_id = self.config["pose_frame_id"]
        p.pose.position.x = t_PB[0]
        p.pose.position.y = t_PB[1]
        p.pose.position.z = t_PB[2]
        p.pose.orientation.x = q_PB.x
        p.pose.orientation.y = q_PB.y
        p.pose.orientation.z = q_PB.z
        p.pose.orientation.w = q_PB.w
        return p

    def _rgb_to_msg(self, observation: Observation) -> Image:
        """Convert the RGB image from the observation to a ROS Image message."""
        msg = self._bridge.cv2_to_imgmsg(observation["rgb"], "rgb8")
        msg.header.stamp = observation["timestamp"]
        return msg

    def _depth_to_msg(self, observation: Observation) -> Image:
        """Convert the depth image from the observation to a ROS Image
        message."""
        msg = self._bridge.cv2_to_imgmsg(observation["depth"], "32FC1")
        msg.header.stamp = observation["timestamp"]
        return msg

    def _sem_instances_to_msg(self, observation: Observation) -> Image:
        """Convert the instance ID image from the observation to a ROS Image
        message."""
        # Habitat-Sim produces 16-bit per-pixel instance ID images.
        msg = self._bridge.cv2_to_imgmsg(
            observation["sem_instances"].astype(np.uint16), "16UC1"
        )
        msg.header.stamp = observation["timestamp"]
        return msg

    def _sem_classes_to_msg(self, observation: Observation) -> Image:
        """Convert the class ID image from the observation to a ROS Image
        message."""
        # Habitat-Sim produces 8-bit per-pixel class ID images.
        msg = self._bridge.cv2_to_imgmsg(
            observation["sem_classes"].astype(np.uint8), "8UC1"
        )
        msg.header.stamp = observation["timestamp"]
        return msg

    def _render_sem_instances_to_msg(self, observation: Observation) -> Image:
        """Visualize an instance ID image to a ROS Image message with
        per-instance colours."""
        color_img = self.class_colors[
            observation["sem_instances"] % len(self.class_colors)
        ]
        color_img = color_img / 2 + observation["rgb"] / 2
        msg = self._bridge.cv2_to_imgmsg(color_img.astype(np.uint8), "rgb8")
        msg.header.stamp = observation["timestamp"]
        return msg

    def _render_sem_classes_to_msg(self, observation: Observation) -> Image:
        """Visualize a class ID image to a ROS Image message with per-class
        colours."""
        color_img = self.class_colors[
            observation["sem_classes"] % len(self.class_colors)
        ]
        color_img = color_img / 2 + observation["rgb"] / 2
        msg = self._bridge.cv2_to_imgmsg(color_img.astype(np.uint8), "rgb8")
        msg.header.stamp = observation["timestamp"]
        return msg

    def _camera_intrinsics_to_msg(self, config: Config) -> CameraInfo:
        """Return a ROS message containing the Habitat-Sim camera intrinsic
        parameters."""
        msg = CameraInfo()
        msg.width = config["width"]
        msg.height = config["height"]
        msg.k = config["K"].flatten().tolist()
        msg.p = config["P"].flatten().tolist()
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        return msg

    def _T_IC_to_T_HB(self, T_IC: np.array) -> np.array:
        """Convert T_IC to T_HB."""
        return self._T_HI @ T_IC @ self._T_CB

    def _T_HB_to_T_IC(self, T_HB: np.array) -> np.array:
        """Convert T_HB to T_IC."""
        return self._T_IH @ T_HB @ self._T_BC

    def _move_and_render(self, sim: Sim, config: Config) -> Observation:
        """Move the habitat sensor and return its observations and ground truth
        pose."""
        # Receive the latest pose.
        self.T_HB_mutex.acquire()
        T_HB = np.copy(self.T_HB)
        stamp = self.T_HB_stamp
        T_HB_received = self.T_HB_received
        self.T_HB_received = False
        self.T_HB_mutex.release()
        # Move the sensor to the pose contained in self.T_HB.
        t_IC, q_IC = split_pose(self._T_HB_to_T_IC(T_HB))
        agent_state = hs.agent.AgentState(t_IC, q_IC)
        self.sim.get_agent(0).set_state(agent_state)
        # Render the sensor observations.
        observation = sim.get_sensor_observations()
        if T_HB_received:
            # Set the observation timestamp to that of the received pose to keep
            # them in sync.
            observation["timestamp"] = stamp
        else:
            # No new pose received yet, use the current timestamp.
            observation["timestamp"] = self.get_clock().now().to_msg()
        # Change from RGBA to RGB
        observation["rgb"] = observation["rgb"][..., 0:3]
        if config["enable_semantics"] and config["instance_to_class"].size > 0:
            # Assuming the scene has no more than 65534 objects
            observation["sem_instances"] = np.clip(
                observation["semantic"].astype(np.uint16), 0, 65535
            )
            del observation["semantic"]
            # Convert instance IDs to class IDs
            observation["sem_classes"] = np.array(
                [config["instance_to_class"][x] for x in observation["sem_instances"]],
                dtype=np.uint8,
            )
        # Get the camera ground truth pose (T_IC) in the habitat frame from the
        # position and orientation
        t_IC = sim.get_agent(0).get_state().position
        q_IC = sim.get_agent(0).get_state().rotation
        T_IC = combine_pose(t_IC, q_IC)
        observation["T_HB"] = self._T_IC_to_T_HB(T_IC)
        return observation

    def _publish_observation(
        self, obs: Observation, pub: Publishers, config: Config
    ) -> None:
        """Publish the sensor observations and ground truth pose."""
        # Broadcast dynamic TF world -> robot
        self._broadcast_tf(obs["T_HB"])
        # Publish messages
        pub["pose"].publish(self._pose_to_msg(obs))
        pub["rgb"].publish(self._rgb_to_msg(obs))
        pub["depth"].publish(self._depth_to_msg(obs))
        if config["enable_semantics"] and config["instance_to_class"].size > 0:
            if config["allowed_classes"]:
                self._filter_sem_classes(obs)
            pub["sem_class"].publish(self._sem_classes_to_msg(obs))
            pub["sem_instance"].publish(self._sem_instances_to_msg(obs))
            # Publish semantics visualisations
            if config["visualize_semantics"]:
                pub["sem_class_render"].publish(self._render_sem_classes_to_msg(obs))
                pub["sem_instance_render"].publish(
                    self._render_sem_instances_to_msg(obs)
                )

    def _record_observation(self, obs: Observation, recording_dir: str) -> None:
        os.makedirs(recording_dir, exist_ok=True)
        os.makedirs(recording_dir + "/depth", exist_ok=True)
        os.makedirs(recording_dir + "/rgb", exist_ok=True)
        stamp_str = "{:.7f}".format(obs["timestamp"].to_sec())
        # Update groundtruth.txt. Write the header if needed.
        groundtruth_txt = recording_dir + "/groundtruth.txt"
        if not os.path.isfile(groundtruth_txt):
            with open(groundtruth_txt, "w") as f:
                f.write("# ground truth trajectory\n")
                f.write("# timestamp tx ty tz qx qy qz qw\n")
        with open(groundtruth_txt, "a") as f:
            T_PH = find_tf(self.tf_buffer, self.config["pose_frame_id"], "habitat")
            t_PC, q_PC = split_pose(T_PH @ obs["T_HB"] @ self._T_BCtum)
            f.write(
                "{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
                    stamp_str, t_PC[0], t_PC[1], t_PC[2], q_PC.x, q_PC.y, q_PC.z, q_PC.w
                )
            )
        # Update depth.txt and rgb.txt. Write the header if needed.
        for t in ["depth", "rgb"]:
            type_txt = "".join([recording_dir, "/", t, ".txt"])
            image_png = "".join([t, "/", stamp_str, ".png"])
            if not os.path.isfile(type_txt):
                with open(type_txt, "w") as f:
                    f.write("# {} images\n".format(t))
                    f.write("# timestamp filename\n")
            with open(type_txt, "a") as f:
                f.write("{} {}\n".format(stamp_str, image_png))
        # Write the depth image. Set big float values to 0 so that it fits in a
        # TUM PNG.
        depth_png = "".join([recording_dir, "/depth/", stamp_str, ".png"])
        depth_constrained = obs["depth"].astype(np.float32)
        depth_constrained[depth_constrained < 0] = 0
        depth_constrained[depth_constrained >= (2**16 - 1) / 5000] = 0
        cv2.imwrite(depth_png, (5000 * depth_constrained).astype(np.uint16))
        # Write the RGB image.
        rgb_png = "".join([recording_dir, "/rgb/", stamp_str, ".png"])
        cv2.imwrite(rgb_png, cv2.cvtColor(obs["rgb"], cv2.COLOR_BGR2RGB))


def main() -> None:
    """Start a node."""
    rclpy.init()

    node = None
    try:
        node = HabitatROSNode()
        habitat_ros.setup_ros_log_forwarding(node)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
