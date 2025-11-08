#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import time
from queue import Queue
from typing import Any

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

# from ..teleoperator import Teleoperator
# from .configuration_keyboard import KeyboardEndEffectorTeleopConfig, KeyboardTeleopConfig
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardEndEffectorTeleopConfig, KeyboardTeleopConfig
from lerobot.teleoperators.teleoperator import Teleoperator

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")


# class KeyboardTeleop(Teleoperator):
#     """
#     Teleop class to use keyboard inputs for control.
#     """

#     config_class = KeyboardTeleopConfig
#     name = "keyboard"

#     def __init__(self, config: KeyboardTeleopConfig):
#         super().__init__(config)
#         self.config = config
#         self.robot_type = config.type

#         self.event_queue = Queue()
#         self.current_pressed = {}
#         self.listener = None
#         self.logs = {}

#     @property
#     def action_features(self) -> dict:
#         return {
#             "dtype": "float32",
#             "shape": (len(self.arm),),
#             "names": {"motors": list(self.arm.motors)},
#         }

#     @property
#     def feedback_features(self) -> dict:
#         return {}

#     @property
#     def is_connected(self) -> bool:
#         return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

#     @property
#     def is_calibrated(self) -> bool:
#         pass

#     def connect(self) -> None:
#         if self.is_connected:
#             raise DeviceAlreadyConnectedError(
#                 "Keyboard is already connected. Do not run `robot.connect()` twice."
#             )

#         if PYNPUT_AVAILABLE:
#             logging.info("pynput is available - enabling local keyboard listener.")
#             self.listener = keyboard.Listener(
#                 on_press=self._on_press,
#                 on_release=self._on_release,
#             )
#             self.listener.start()
#         else:
#             logging.info("pynput not available - skipping local keyboard listener.")
#             self.listener = None

#     def calibrate(self) -> None:
#         pass

#     def _on_press(self, key):
#         if hasattr(key, "char"):
#             self.event_queue.put((key.char, True))

#     def _on_release(self, key):
#         if hasattr(key, "char"):
#             self.event_queue.put((key.char, False))
#         if key == keyboard.Key.esc:
#             logging.info("ESC pressed, disconnecting.")
#             self.disconnect()

#     def _drain_pressed_keys(self):
#         while not self.event_queue.empty():
#             key_char, is_pressed = self.event_queue.get_nowait()
#             self.current_pressed[key_char] = is_pressed

#     def configure(self):
#         pass

#     def get_action(self) -> dict[str, Any]:
#         before_read_t = time.perf_counter()

#         if not self.is_connected:
#             raise DeviceNotConnectedError(
#                 "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
#             )

#         self._drain_pressed_keys()

#         # Generate action based on current key states
#         action = {key for key, val in self.current_pressed.items() if val}
#         self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

#         return dict.fromkeys(action, None)

#     def send_feedback(self, feedback: dict[str, Any]) -> None:
#         pass

#     def disconnect(self) -> None:
#         if not self.is_connected:
#             raise DeviceNotConnectedError(
#                 "KeyboardTeleop is not connected. You need to run `robot.connect()` before `disconnect()`."
#             )
#         if self.listener is not None:
#             self.listener.stop()

# 顶部 import 补充
import select
import termios
import tty
import threading
import random
import numpy as np

class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

        # === 新增：状态&线程 ===
        # 关节数量尽量从 self.arm 推断；不可用则默认为 6
        try:
            self.n_joints = len(self.arm)
        except Exception:
            self.n_joints = 7

        self.selected = 0                  # 当前选中的关节索引
        self.step = 5.0                    # 默认步长（度）
        self.joint_pos = [0.0] * self.n_joints
        self.running = False               # stdin 循环运行标志
        self.cli_thread: threading.Thread | None = None

        logging.info(f"[KeyboardTeleop] init: n_joints={self.n_joints}, step={self.step}")

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        # 这里可根据你的实际流程调整；先返回 True
        return True

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "Keyboard is already connected. Do not run `robot.connect()` twice."
            )

        if PYNPUT_AVAILABLE:
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logging.info("pynput not available - skipping local keyboard listener.")
            self.listener = None

        # 启动终端 stdin 循环（两条通道可共存，谁来都能用）
        self.start_cli_loop()

    def calibrate(self) -> None:
        pass

    # ====== pynput 回调：把字符放入队列 + 直接进行事件处理 ======
    def _on_press(self, key):
        try:
            if hasattr(key, "char") and key.char is not None:
                ch = key.char
                logging.debug(f"[pynput] press char={ch!r}")
                self.event_queue.put((ch, True))
                self.handle_key(ch)
            else:
                # 非字符键（方向键/功能键）只入队状态，留给 EE 版本用
                logging.debug(f"[pynput] press non-char key={key}")
                self.event_queue.put((key, True))
        except Exception as e:
            logging.exception(f"_on_press error: {e}")

    def _on_release(self, key):
        try:
            if hasattr(key, "char") and key.char is not None:
                ch = key.char
                logging.debug(f"[pynput] release char={ch!r}")
                self.event_queue.put((ch, False))
            else:
                logging.debug(f"[pynput] release non-char key={key}")
                self.event_queue.put((key, False))

            # ESC 退出
            try:
                if key == keyboard.Key.esc:
                    logging.info("ESC pressed, disconnecting.")
                    self.disconnect()
            except Exception:
                # keyboard 可能为 None 时的兼容
                pass
        except Exception as e:
            logging.exception(f"_on_release error: {e}")

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_obj, is_pressed = self.event_queue.get_nowait()
            # 对于 pynput 非字符键，用其对象作为键；字符键用字符
            k = key_obj
            self.current_pressed[k] = is_pressed

    def configure(self):
        pass

    # ====== 新增：终端 stdin 循环 ======
    def start_cli_loop(self):
        """启动一个线程，在无 GUI / 远程 SSH 情况下仍可读取键盘输入"""
        if self.cli_thread and self.cli_thread.is_alive():
            return
        self.running = True
        self.cli_thread = threading.Thread(target=self._stdin_loop, name="KeyboardTeleopCLI", daemon=False)
        self.cli_thread.start()
        logging.info("[KeyboardTeleop] CLI stdin loop started")
        # 主循环：打印状态
        # from time import sleep
        # while self.running:
        #     print("关节角度:")
        #     sleep(1)

    def stop_cli_loop(self):
        self.running = False
        if self.cli_thread and self.cli_thread.is_alive():
            self.cli_thread.join(timeout=1.0)
        logging.info("[KeyboardTeleop] CLI stdin loop stopped")

    def _stdin_loop(self):

        """原始输入模式读取单字符，并复用 handle_key()"""
        try:
            fd = sys.stdin.fileno()
        except Exception:
            logging.warning("[KeyboardTeleop] stdin not available; CLI loop disabled")
            return

        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self.running:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch:
                        logging.debug(f"[stdin] read char={ch!r}")
                        # 也投递到事件队列，保持与 pynput 一致
                        self.event_queue.put((ch, True))
                        self.handle_key(ch)
        except Exception as e:
            logging.exception(f"[KeyboardTeleop] _stdin_loop error: {e}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    # ====== 新增：你的事件处理逻辑（数字选关节、j/k 移动、[/] 步长、r/z、q）======
    def handle_key(self, ch: str):
        """处理单字符事件，并打印/记录调试信息"""
        try:
            if ch == 'q':
                msg = "退出程序 (q)"
                print(msg)
                logging.info(msg)
                self.running = False
                # 若需要同时断开 pynput 监听，可主动 disconnect()
                return

            if ch in '1234567':
                self.selected = min(max(int(ch) - 1, 0), self.n_joints - 1)
                msg = f"选择关节 {self.selected + 1}"
                print(msg)
                logging.info(msg)
                return

            if ch == 'j':
                self.joint_pos[self.selected] -= self.step
                self.move_joint()
                msg = f"[move] 关节{self.selected + 1} -= {self.step:.2f}°, 当前: {self.joint_pos[self.selected]:.2f}°"
                print(msg)
                logging.info(msg)
                return

            if ch == 'k':
                self.joint_pos[self.selected] += self.step
                self.move_joint()
                msg = f"[move] 关节{self.selected + 1} += {self.step:.2f}°, 当前: {self.joint_pos[self.selected]:.2f}°"
                print(msg)
                logging.info(msg)
                return

            if ch == '[':
                # 步长减小，但不小于 1°
                self.step = max(1.0, self.step / 1.5)
                msg = f"步长减小: {self.step:.2f}°"
                print(msg)
                logging.info(msg)
                return

            if ch == ']':
                # 步长增大，但不超过 30°
                self.step = min(30.0, self.step * 1.5)
                msg = f"步长增大: {self.step:.2f}°"
                print(msg)
                logging.info(msg)
                return

            if ch == 'r':
                self.joint_pos = [0.0] * self.n_joints
                self.move_joint()
                msg = "重置所有关节为 0°"
                print(msg)
                logging.info(msg)
                return

            if ch == 'z':
                self.joint_pos = [random.uniform(-30, 30) for _ in range(self.n_joints)]
                self.move_joint()
                msg = f"随机化关节角度: {np.round(self.joint_pos, 2)}"
                print(msg)
                logging.info(msg)
                return

            # 其他字符：只做调试记录
            logging.debug(f"[handle_key] ignored char: {ch!r}")

        except Exception as e:
            logging.exception(f"[handle_key] error with ch={ch!r}: {e}")

    def move_joint(self):
        """
        实际的关节控制占位实现：
        - 这里只打印/记录；你可以改为发送给机器人驱动/SDK。
        - 同时把当前位置写进 self.logs，便于外部抓取。
        """
        # TODO: 替换为真实控制调用，如 self.arm.move_to(self.joint_pos) 等
        logging.info(f"[move_joint] joint_pos={np.round(self.joint_pos, 2).tolist()}")
        self.logs["joint_pos"] = list(self.joint_pos)

    # def get_action(self) -> dict[str, Any]:
    #     before_read_t = time.perf_counter()

    #     if not self.is_connected and not self.running:
    #         # 既没有 pynput 监听也没有 CLI 循环
    #         raise DeviceNotConnectedError(
    #             "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
    #         )

    #     self._drain_pressed_keys()

    #     # 仅保留按下中的键集合（pynput 情况下可能有用）
    #     action_keys = {k for k, v in self.current_pressed.items() if v}
    #     self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

    #     # 返回值按你的原设计：键集合 -> None
    #     action_dict = dict.fromkeys(action_keys, None)

    #     # 同时把我们维护的关节位姿一起反馈（便于你的上层取用/调试）
    #     action_dict["_selected_joint"] = self.selected
    #     action_dict["_step_deg"] = self.step
    #     action_dict["_joint_pos_deg"] = list(self.joint_pos)
    #     print(f"action_dict: {action_dict}")
    #     # 清空一次性状态（可选）
    #     self.current_pressed.clear()
    #     return action_dict
    def get_action(self) -> dict[str, Any]:
        before_read_t = time.perf_counter()

        if not self.is_connected and not self.running:
            # 既没有 pynput 监听也没有 CLI 循环
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        # 仅保留按下中的键集合（pynput 情况下可能有用）
        action_keys = {k for k, v in self.current_pressed.items() if v}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        # -------------------------------
        # 1. 按键事件：统一命名为 "key.xxx"
        # -------------------------------
        key_actions = {f"key.{k}": None for k in action_keys}

        # -------------------------------
        # 2. 关节角度：统一命名为 "joint_i.pos"
        # -------------------------------
        joint_actions = {f"joint_{i+1}.pos": val
                        for i, val in enumerate(self.joint_pos)}

        # -------------------------------
        # 3. 额外调试字段：也规范化
        # -------------------------------
        meta_actions = {
            "meta.selected_joint": self.selected,
            "meta.step_deg": self.step,
            "meta.joint_pos_deg": list(self.joint_pos),
        }

        # 合并所有结果
        action_dict = {}
        # action_dict.update(key_actions)
        action_dict.update(joint_actions)
        # action_dict.update(meta_actions)

        print(f"[get_action] action_dict={action_dict}")
        self.current_pressed.clear()
        return action_dict


    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected and not self.running:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `robot.connect()` before `disconnect()`."
            )
        if self.listener is not None and self.is_connected:
            self.listener.stop()
        self.stop_cli_loop()

class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    def _on_press(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, False))

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()
        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper_action = 1.0

        # Generate action based on current key states
        for key, val in self.current_pressed.items():
            if key == keyboard.Key.up:
                delta_y = -int(val)
            elif key == keyboard.Key.down:
                delta_y = int(val)
            elif key == keyboard.Key.left:
                delta_x = int(val)
            elif key == keyboard.Key.right:
                delta_x = -int(val)
            elif key == keyboard.Key.shift:
                delta_z = -int(val)
            elif key == keyboard.Key.shift_r:
                delta_z = int(val)
            elif key == keyboard.Key.ctrl_r:
                # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
                gripper_action = int(val) + 1
            elif key == keyboard.Key.ctrl_l:
                gripper_action = int(val) - 1
            elif val:
                # If the key is pressed, add it to the misc_keys_queue
                # this will record key presses that are not part of the delta_x, delta_y, delta_z
                # this is useful for retrieving other events like interventions for RL, episode success, etc.
                self.misc_keys_queue.put(key)

        self.current_pressed.clear()

        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
        }

        if self.config.use_gripper:
            action_dict["gripper"] = gripper_action

        return action_dict

if __name__ == "__main__":

    from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,    
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
    keyboard
)

    
    
    teleop_cfg = keyboard.KeyboardTeleopConfig(
    id="my_awesome_leader_arm",
)
    from lerobot.teleoperators.utils import make_teleoperator_from_config
    teleop = make_teleoperator_from_config(teleop_cfg) if teleop_cfg is not None else None
    teleop.connect()