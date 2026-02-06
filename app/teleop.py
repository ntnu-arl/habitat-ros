#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020-2021 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020-2021 Sotiris Papatheodorou
# BSD 3-Clause License

# Copyright (c) 2026, NTNU Autonomous Robots Lab
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
"""Teleoperation node for moving a robot by keyboard input."""

import curses
import math
import time
from typing import Tuple

import numpy as np
import quaternion
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node

_node_name = "teleop"
_pose_input_topic = "/habitat/pose"
_output_topic = "/habitat/external_pose"


class Movement:
    _x_step = 0.25
    _y_step = 0.25
    _z_step = 0.25
    _roll_step = 5
    _pitch_step = 5
    _yaw_step = 5

    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
        self._x = x
        self._y = y
        self._z = z
        self._roll = roll
        self._pitch = pitch
        self._yaw = yaw

    def x(self) -> float:
        return self._x * Movement._x_step

    def y(self) -> float:
        return self._y * Movement._y_step

    def z(self) -> float:
        return self._z * Movement._z_step

    def roll(self) -> float:
        return math.radians(self._roll * Movement._roll_step)

    def pitch(self) -> float:
        return math.radians(self._pitch * Movement._pitch_step)

    def yaw(self) -> float:
        return math.radians(self._yaw * Movement._yaw_step)


class TeleopNode(Node):
    def __init__(self):
        super().__init__(_node_name)

        # ROS2 parameter
        self.declare_parameter("publish_path", False)
        publish_path = self.get_parameter("publish_path").value
        self.publish_path = publish_path

        # Publisher
        if publish_path:
            self.pub = self.create_publisher(Path, _output_topic, 10)
        else:
            self.pub = self.create_publisher(PoseStamped, _output_topic, 10)

        # Pose storage
        self.pose_msg = None

        self.get_logger().info("Teleop node initialized. Waiting for initial pose...")

        # Subscribe once to get the initial pose
        self.init_pose()
        self.get_logger().info("Initial pose received. Ready for teleoperation.")

    def init_pose(self):
        # Synchronous subscriber to get the first PoseStamped
        self.pose_msg = None

        def callback(msg):
            self.pose_msg = msg

        sub = self.create_subscription(PoseStamped, _pose_input_topic, callback, 10)

        while rclpy.ok() and self.pose_msg is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.destroy_subscription(sub)

    def pose_from_str(self, p: PoseStamped, s: str) -> PoseStamped:
        new_p = PoseStamped()
        new_p.header.stamp = self.get_clock().now().to_msg()
        new_p.header.frame_id = p.header.frame_id
        e = s.split()
        if len(e) != 7:
            self.get_logger().fatal(
                f"Invalid TSV line, expected 7 columns, got {len(e)}\n  {s}"
            )
            raise KeyboardInterrupt
        new_p.pose.position.x = float(e[0])
        new_p.pose.position.y = float(e[1])
        new_p.pose.position.z = float(e[2])
        new_p.pose.orientation.x = float(e[3])
        new_p.pose.orientation.y = float(e[4])
        new_p.pose.orientation.z = float(e[5])
        new_p.pose.orientation.w = float(e[6])
        return new_p

    def update_pose(self, p: PoseStamped, m: Movement) -> PoseStamped:
        new_p = PoseStamped()
        new_p.header.stamp = self.get_clock().now().to_msg()
        new_p.header.frame_id = p.header.frame_id

        # Current pose matrix
        T_HB = np.identity(4)
        T_HB[0, 3] = p.pose.position.x
        T_HB[1, 3] = p.pose.position.y
        T_HB[2, 3] = p.pose.position.z
        q_current = quaternion.quaternion(
            p.pose.orientation.w,
            p.pose.orientation.x,
            p.pose.orientation.y,
            p.pose.orientation.z,
        )
        T_HB[0:3, 0:3] = quaternion.as_rotation_matrix(q_current)

        # Movement in body frame
        T_BBnew = np.identity(4)
        T_BBnew[0, 3] += m.x()
        T_BBnew[1, 3] += m.y()
        T_BBnew[2, 3] += m.z()

        q_yaw = quaternion.quaternion(
            math.cos(m.yaw() / 2), 0, 0, math.sin(m.yaw() / 2)
        )
        q_pitch = quaternion.quaternion(
            math.cos(m.pitch() / 2), 0, math.sin(m.pitch() / 2), 0
        )
        q_roll = quaternion.quaternion(
            math.cos(m.roll() / 2), math.sin(m.roll() / 2), 0, 0
        )
        q_new = q_yaw * q_pitch * q_roll
        T_BBnew[0:3, 0:3] = quaternion.as_rotation_matrix(q_new)

        # New pose
        T_HBnew = T_HB @ T_BBnew
        q = quaternion.from_rotation_matrix(T_HBnew[0:3, 0:3])
        new_p.pose.position.x = T_HBnew[0, 3]
        new_p.pose.position.y = T_HBnew[1, 3]
        new_p.pose.position.z = T_HBnew[2, 3]
        new_p.pose.orientation.x = q.x
        new_p.pose.orientation.y = q.y
        new_p.pose.orientation.z = q.z
        new_p.pose.orientation.w = q.w
        return new_p

    def pose_to_path(self, pose: PoseStamped, new_pose: PoseStamped) -> Path:
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = pose.header.frame_id
        path.poses.append(pose)
        path.poses.append(new_pose)
        return path

    def wait_for_key(self, window) -> Tuple[Movement, bool]:
        m = Movement()
        quit = False

        while True:
            try:
                key = window.getkey()
            except KeyboardInterrupt:
                # Let outer loop handle shutdown
                quit = True
                break

            if key == "Q":
                quit = True
                break
            elif key == "w":
                m = Movement(x=1)
                break
            elif key == "s":
                m = Movement(x=-1)
                break
            elif key == "a":
                m = Movement(y=1)
                break
            elif key == "d":
                m = Movement(y=-1)
                break
            elif key == " ":
                m = Movement(z=1)
                break
            elif key == "c":
                m = Movement(z=-1)
                break
            elif key == "q":
                m = Movement(yaw=1)
                break
            elif key == "e":
                m = Movement(yaw=-1)
                break
            elif key == "f":
                m = Movement(pitch=1)
                break
            elif key == "r":
                m = Movement(pitch=-1)
                break
            elif key == "x":
                m = Movement(roll=1)
                break
            elif key == "z":
                m = Movement(roll=-1)
                break

            time.sleep(0.05)

        return m, quit

    def print_waiting_for_pose(self, window):
        window.clear()
        window.addstr(
            1,
            0,
            f"Waiting for initial pose on topic {_pose_input_topic}",
        )
        window.refresh()

    def print_help(self, window):
        window.addstr(0, 0, "Position:")
        window.addstr(2, 0, "Orientation (w,x,y,z):")
        window.addstr(5, 0, "w/s       forwards/backwards")
        window.addstr(6, 0, "a/d       left/right")
        window.addstr(7, 0, "space/c   up/down")
        window.addstr(8, 0, "q/e       yaw left/right")
        window.addstr(9, 0, "r/f       pitch up/down")
        window.addstr(10, 0, "z/x       roll CCW/CW")
        window.addstr(11, 0, "Q         quit")

    def print_pose_stamped(self, p: PoseStamped, window):
        position = [p.pose.position.x, p.pose.position.y, p.pose.position.z]
        orientation = [
            p.pose.orientation.w,
            p.pose.orientation.x,
            p.pose.orientation.y,
            p.pose.orientation.z,
        ]
        window.move(1, 0)
        window.clrtoeol()
        window.addstr(1, 0, "  " + " ".join(["{: 8.3f}".format(x) for x in position]))
        window.move(3, 0)
        window.clrtoeol()
        window.addstr(
            3, 0, "  " + " ".join(["{: 8.3f}".format(x) for x in orientation])
        )

    def run(self):
        window = None
        curses_active = False
        try:
            window = curses.initscr()
            curses_active = True

            curses.noecho()
            curses.cbreak()
            window.keypad(True)

            self.print_waiting_for_pose(window)
            pose = self.pose_msg
            quit = False
            self.print_help(window)

            while rclpy.ok() and not quit:
                self.print_pose_stamped(pose, window)
                movement, quit = self.wait_for_key(window)

                if quit:
                    break

                new_pose = self.update_pose(pose, movement)
                if self.publish_path:
                    self.pub.publish(self.pose_to_path(pose, new_pose))
                else:
                    self.pub.publish(new_pose)
                pose = new_pose

        except KeyboardInterrupt:
            # Clean exit on Ctrl-C
            pass

        finally:
            if curses_active:
                try:
                    curses.nocbreak()
                    window.keypad(False)
                    curses.echo()
                    curses.endwin()
                except curses.error:
                    # Already closed by signal handler – ignore
                    pass


def main():
    rclpy.init()
    node = TeleopNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
