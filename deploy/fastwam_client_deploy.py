import argparse
import logging
import time
import threading
import sys
import select
import termios
import tty
from collections import deque
import numpy as np
from typing import Any, Tuple, Optional
from deploy_common import EMA, RealEnv, RIGHT_ARM_POLICY_DIM
from openpi_client.websocket_client_policy import WebsocketClientPolicy


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class KeyboardEEIntervention:
    """键盘末端 XYZ 平移人工介入控制器（无旋转控制）。

    按键映射:
        平移:  w/s = +/-X,  a/d = +/-Y,  q/e = +/-Z
        夹爪:  g = 打开/关闭切换
        步长:  +/- = 增大/减小步长
        切换:  SPACE = 启用介入，x = 结束介入（交回 policy 控制）
        退出:  ESC = 退出当前 episode

    工作流:
        1. policy.infer() 得到 action chunk
        2. SPACE 启用介入后，每步将 XYZ delta 以 ik_use=True 发给 env.step()
        3. env.step() 首次 IK 步 switch_control_mode(wr_wbc_right_ik)，
           get_current_tcp_poses + delta → send_action_right；按 x 后下一步
           env.step(ik_use=False) 切回 xr_wbc 并 send_action
        4. 介入步 step 后：用 whole_body IK 仅合并覆盖本步 VLA 的右手 8 维并重置 EMA；
           action_queue 后续步仍为原 VLA chunk。ESC 退出 episode
    """

    # 按键 → (轴索引, 方向)  轴: 0=X, 1=Y, 2=Z（仅平移）
    KEY_MAP = {
        'w': (0, +1), 's': (0, -1),   # X
        'a': (1, +1), 'd': (1, -1),   # Y
        'q': (2, +1), 'e': (2, -1),   # Z
    }

    def __init__(
        self,
        *,
        linear_step: float = 0.005,
        step_scale_factor: float = 2.0,
    ) -> None:
        """
        Args:
            linear_step:  每次按键的平移增量 (m)
            step_scale_factor: 按 +/- 时步长的缩放因子
        """
        self.linear_step = linear_step
        self.step_scale_factor = step_scale_factor

        self._is_active = False          # 是否正在人工介入
        self._should_quit = False        # ESC 请求退出 episode
        self._gripper_toggle = False     # 本帧是否切换夹爪
        self._delta = np.zeros(3, dtype=np.float64)   # XYZ 增量
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ---- 公开属性 ----
    @property
    def is_active(self) -> bool:
        """True 时表示人工介入已启用。"""
        return self._is_active

    @property
    def should_quit(self) -> bool:
        """True 时应退出当前 episode。"""
        return self._should_quit

    # ---- 生命周期 ----
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._should_quit = False
        self._is_active = False
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logging.info(
            "KeyboardEEIntervention 已启动: SPACE 开始介入, w/s/a/d/q/e 平移, "
            "g 夹爪, +/- 步长, x 结束介入, ESC 退出"
        )

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_delta(self) -> Tuple[np.ndarray, bool]:
        """获取并清零本帧的 XYZ 介入增量。

        Returns:
            delta: shape=(3,) 末端平移增量 [dx, dy, dz] (m)
            gripper_toggle: 是否切换夹爪
        """
        with self._lock:
            delta = self._delta.copy()
            gripper = self._gripper_toggle
            self._delta[:] = 0.0
            self._gripper_toggle = False
        return delta, gripper

    def has_pending_delta(self) -> bool:
        """是否有待处理的非零 delta（不清零，仅查看）。"""
        with self._lock:
            return np.any(self._delta != 0.0) or self._gripper_toggle

    # ---- 内部：键盘监听 ----
    def _listen_loop(self) -> None:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self._running:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    self._handle_key(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_key(self, ch: str) -> None:
        # ESC
        if ch == '\x1b':
            self._should_quit = True
            logging.info("[Intervention] ESC → 请求退出 episode")
            return

        # SPACE 启用介入
        if ch == ' ':
            if not self._is_active:
                self._is_active = True
                logging.info("[Intervention] 人工介入已启用 (按 x 结束)")
            return

        # x 键结束介入，交回 policy
        if ch == 'x':
            if self._is_active:
                self._is_active = False
                # 清零残留 delta
                with self._lock:
                    self._delta[:] = 0.0
                    self._gripper_toggle = False
                logging.info("[Intervention] 人工介入已结束，恢复 policy 控制")
            return

        if not self._is_active:
            return

        # 步长调整
        if ch in ('+', '='):
            self.linear_step *= self.step_scale_factor
            logging.info(f"[Intervention] 步长增大 → {self.linear_step:.4f} m")
            return
        if ch in ('-', '_'):
            self.linear_step /= self.step_scale_factor
            logging.info(f"[Intervention] 步长减小 → {self.linear_step:.4f} m")
            return

        # 夹爪切换
        if ch == 'g':
            with self._lock:
                self._gripper_toggle = True
            logging.info("[Intervention] 夹爪切换")
            return

        # XYZ 平移增量
        if ch in self.KEY_MAP:
            axis, direction = self.KEY_MAP[ch]
            with self._lock:
                self._delta[axis] += direction * self.linear_step
            # logging.info(
            #     f"[Intervention] delta_xyz = [{self._delta[0]:+.4f}, "
            #     f"{self._delta[1]:+.4f}, {self._delta[2]:+.4f}]"
            # )


class WholeBodyCmdReader:
    """订阅 whole_body/whole_body_data，缓存最新的 IK 解 joint_state_cmd。

    用于键盘介入后读取 IK 解算结果，覆盖当前步 action。

    action 格式 (19-dim, 无底盘):
        [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1), torso(3)]
    """

    def __init__(self, topic: str = "whole_body/whole_body_data"):
        """初始化 WholeBodyCmdReader。

        Args:
            topic: WholeBodyData 话题名
        """
        self._topic = topic
        self._latest_cmd: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._sub = None
        # NOTE: 需要在 rclpy 已初始化后、且有 Node 可用时调用 attach_to_node()

    def attach_to_node(self, node) -> None:
        """将订阅者挂载到已有的 ROS2 节点上。

        Args:
            node: rclpy.node.Node 实例（通常为 env.robot）
        """
        from x_msgs.msg import WholeBodyData
        self._sub = node.create_subscription(
            WholeBodyData,
            self._topic,
            self._callback,
            10,
        )
        logging.info(f"WholeBodyCmdReader 已订阅 '{self._topic}'")

    def _callback(self, msg) -> None:
        """解析 WholeBodyData，提取各 ControlGroup 的 joint_state_cmd。

        ControlGroupData 的 name 字段标识组名（如 'left_arm', 'right_arm', 'body'）。
        joint_state_cmd.position 即为 IK 解算后的关节指令。
        """
        cmd_dict: dict[str, np.ndarray] = {}
        for cg in msg.data:
            name = cg.name.lower()
            if cg.joint_state_cmd and cg.joint_state_cmd.position:
                cmd_dict[name] = np.array(cg.joint_state_cmd.position, dtype=np.float32)

        # TODO: 根据你的实际 ControlGroup name 映射调整以下键名
        # 常见: 'left_arm', 'right_arm', 'body'/'torso'
        left_arm_cmd = cmd_dict.get("left_arm", np.zeros(7, dtype=np.float32))
        right_arm_cmd = cmd_dict.get("right_arm", np.zeros(7, dtype=np.float32))
        torso_cmd = cmd_dict.get("body", cmd_dict.get("torso", np.zeros(3, dtype=np.float32)))

        # 夹爪 cmd 从 gripper_joint_state_cmd 读取
        left_gripper_cmd = 0.0
        right_gripper_cmd = 0.0
        for cg in msg.data:
            name = cg.name.lower()
            if name == "left_arm" and cg.gripper_joint_state_cmd and cg.gripper_joint_state_cmd.position:
                left_gripper_cmd = float(cg.gripper_joint_state_cmd.position[0]) if cg.gripper_joint_state_cmd.position else 0.0
            if name == "right_arm" and cg.gripper_joint_state_cmd and cg.gripper_joint_state_cmd.position:
                right_gripper_cmd = float(cg.gripper_joint_state_cmd.position[0]) if cg.gripper_joint_state_cmd.position else 0.0

        # 按 19-dim 格式组装（无底盘）: [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1), torso(3)]
        action_cmd = np.concatenate([
            left_arm_cmd[:7],
            [left_gripper_cmd],
            right_arm_cmd[:7],
            [right_gripper_cmd],
            torso_cmd[:3],
        ]).astype(np.float32)

        with self._lock:
            self._latest_cmd = action_cmd

    def get_latest_cmd(self) -> Optional[np.ndarray]:
        """获取最新的 IK 解 joint cmd (19-dim, 无底盘)，若尚未收到则返回 None。"""
        with self._lock:
            return self._latest_cmd.copy() if self._latest_cmd is not None else None


# 与 whole_body / HAND_TEST 一致: [left(7+1), right(7+1), torso(3)]，VLA 仅预测右手 8 维
_RIGHT_ARM_IN_FULL_ENV = slice(8, 16)


def _merge_wholebody_ik_right_arm8(
    vla_row: np.ndarray,
    ik_row: Optional[np.ndarray],
) -> np.ndarray:
    """用 whole_body IK 的右手 8 维（右臂 7 + 右夹爪 1）覆盖 VLA 对应部分，其余维保留 VLA。

    - 若 ``vla_row`` 为完整 env 行（≥16，常见 19/22）：只改 ``[8:16]``。
    - 若 ``vla_row`` 仅为右手 8 维（与 ``RIGHT_ARM_POLICY_DIM`` 一致）：用 ``ik[8:16]`` 作为整行。
    """
    sa = np.asarray(vla_row, dtype=np.float32).ravel()
    out = sa.copy()
    if ik_row is None:
        return out
    ik = np.asarray(ik_row, dtype=np.float32).ravel()
    sl = _RIGHT_ARM_IN_FULL_ENV
    if sa.size == int(RIGHT_ARM_POLICY_DIM):
        if ik.size >= sl.stop:
            return ik[sl].astype(np.float32, copy=True)
        if ik.size >= int(RIGHT_ARM_POLICY_DIM):
            return ik[: int(RIGHT_ARM_POLICY_DIM)].astype(np.float32, copy=True)
        logging.warning(
            "[Intervention] IK 维 %d 不足以提供右手 8 维，保持 VLA 行",
            ik.size,
        )
        return out
    if sa.size >= sl.stop and ik.size >= sl.stop:
        out[sl] = ik[sl]
        return out
    logging.warning(
        "[Intervention] 无法按右手 8 维合并: vla_row=%s ik_row=%s",
        sa.shape,
        ik.shape,
    )
    return out


def _resize_action_vector(vec: np.ndarray, target_dim: int) -> np.ndarray:
    """将任意长度关节向量对齐到 target_dim：短则尾部补零，长则截断。"""
    v = np.asarray(vec, dtype=np.float32).ravel()
    if v.size == target_dim:
        return v.copy()
    if v.size < target_dim:
        o = np.zeros(target_dim, dtype=np.float32)
        o[: v.size] = v
        return o
    if v.size > target_dim:
        logging.warning(
            "[Intervention] 处理后向量维 %d > policy 行维 %d，已截断尾部",
            v.size,
            target_dim,
        )
        return v[:target_dim].copy()
    return v.copy()


class FastWAMClientPolicy:
    """FastWAM 策略客户端 — 发送图像和状态给 fastwam_server.py 服务端，
    接收动作序列（action_horizon, action_dim）。

    特点：
      - 发送格式：observation.image.chest, observation.image.right_wrist, state, instruction
      - 返回格式：{"action": np.ndarray} with shape (action_horizon, action_dim)
      - 无特殊参数校验，直接推理
      - 图像必须是 uint8 格式，形状 [3, 480, 640]
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        image_keys: Tuple[str, ...] = ("chest", "right_wrist"),
        action_horizon: int = 64,
    ) -> None:
        self.client = WebsocketClientPolicy(host=host, port=port)
        self._image_keys = image_keys
        self._action_horizon = action_horizon
        
        # 获取服务器metadata并验证
        meta = self.client.get_server_metadata()
        if meta.get("model") != "FastWAM":
            raise ValueError(
                f"服务端不是 FastWAM 策略（metadata.model={meta.get('model')!r}），"
                "请确认启动的是 fastwam_server.py"
            )
        logging.info(f"Connected to FastWAM Policy Server at {host}:{port}")

    def infer(self, obs: dict[str, Any]) -> np.ndarray:
        """发送观测，返回动作序列。

        Args:
            obs:   机器人观测字典，包含：
                - "chest": np.ndarray [3, 480, 640] uint8 图像
                - "right_wrist": np.ndarray [3, 480, 640] uint8 图像
                - "state": np.ndarray 全关节状态
                - "instruction": str 任务描述

        Returns:
            np.ndarray shape=(action_horizon, action_dim) 的动作序列
        """
        
        # 准备观测字典，使用FastWAM服务器期望的键名
        policy_obs: dict[str, Any] = {
            "observation.image.chest": obs.get("chest").transpose(2, 0, 1).astype(np.uint8),
            "observation.image.right_wrist": obs.get("right_wrist").transpose(2, 0, 1).astype(np.uint8),
            "state": obs.get("state")[8:16].copy()[None, :].astype(np.float32),
            "instruction": "pick and place",
        }

        t0 = time.time()
        response = self.client.infer(policy_obs)
        latency_ms = (time.time() - t0) * 1000
        logging.info(f"FastWAM inference latency: {latency_ms:.2f} ms")

        if "action" not in response:
            raise ValueError(f"Unexpected response from FastWAM server: {response.keys()}")

        action_np = response["action"]
        return action_np  # shape: (action_horizon, action_dim)


# ================================= 运行入口 =================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-robot", action="store_true")
    parser.add_argument("--allow-missing-images", action="store_true",
                        help="相机 topic 未收到数据时，用 640×480 全零 RGB 占位，避免卡住")
    parser.add_argument("--host", default="192.168.21.99")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument(
        "--num-open-loop-steps",
        type=int,
        default=50,
        help="每轮从服务端返回的 action chunk 中取前 N 步执行，建议与服务端的 action_horizon 一致",
    )
    parser.add_argument("--task-description", type=str,
                        default="pick up the box on the table and stack it on the box on the floor",
                        help="任务描述字符串，发送给服务端")
    parser.add_argument("--ema-alpha", type=float, default=1.0,
                        help="FastWAM 模式下动作 EMA 平滑系数，1.0 = 不平滑")
    parser.add_argument(
        "--keyboard-ee-intervention",
        action="store_true",
        help="启用键盘末端介入：SPACE 开始介入，w/s/a/d/q/e XYZ 平移，g 夹爪，+/- 步长，x 结束介入恢复 policy，ESC 退出 episode",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=64,
        help="FastWAM 模式下每次推理返回的动作序列长度",
    )
    args = parser.parse_args()

    HOST = args.host
    PORT = args.port
    NUM_STEPS = args.num_open_loop_steps
    INIT_ROBOT = args.init_robot
    TASK_DESCRIPTION = args.task_description

    # ======================== FastWAM 部署模式 ========================
    # 与 fastwam_server.py 配套使用：
    #   - 客户端发送 chest + right_wrist 图像、原始状态和任务描述
    #   - 服务端完成 VLA 编码 + FastWAM 推理
    #   - 客户端接收完整动作序列 (action_horizon, action_dim)，直接步进机器人
    policy = FastWAMClientPolicy(
        host=HOST,
        port=PORT,
        image_keys=("chest", "right_wrist"),
        action_horizon=64,
    )

    env = RealEnv(
        init_robot=INIT_ROBOT,
        hand_test=False,
        hand_test_no_chest=False,
        allow_missing_images=args.allow_missing_images,
        td3_mode=False,
    )

    # ---- 键盘介入 ----
    keyboard_intervention: Optional[KeyboardEEIntervention] = None
    wb_cmd_reader: Optional[WholeBodyCmdReader] = None
    if args.keyboard_ee_intervention:
        keyboard_intervention = KeyboardEEIntervention(
            linear_step=0.005,
            step_scale_factor=2.0,
        )
        wb_cmd_reader = WholeBodyCmdReader(topic="whole_body/whole_body_data")
        wb_cmd_reader.attach_to_node(env.robot)  # 挂载到机器人 ROS 节点

    episode_idx = 0
    try:
        while True:
            cmd = input("Press Enter to start new episode, or type 'q' to quit... ")
            if cmd.strip().lower() in {"q", "quit", "exit"}:
                break
            
            obs = env.reset()      # FastWAM 模式：重置环境并获取初始观测
            action_queue: deque = deque()
            ema = EMA(alpha=args.ema_alpha)
            t = 0

            if keyboard_intervention is not None:
                keyboard_intervention.start()

            try:
                while t < 50000:
                    # ---- 键盘介入：ESC 退出 ----
                    if keyboard_intervention is not None and keyboard_intervention.should_quit:
                        logging.info("键盘 ESC → 结束当前 episode")
                        break

                    # ---- 获取 action chunk（不管是否介入都需要先推理） ----
                    if len(action_queue) == 0:
                        obs["instruction"] = TASK_DESCRIPTION
                        fastwam_obs = {
                            "chest": obs.get("chest"),
                            "right_wrist": obs.get("right_wrist"),
                            "state": obs.get("state"),
                            "instruction": obs.get("instruction", ""),
                        }
                        raw_actions = policy.infer(fastwam_obs)  # shape: (action_horizon, action_dim)
                        
                        # Padding zero to 0:8 (left arm) and 16:19, raw actions shape: (64, 8), new shape: (64, 19)
                        actions = np.zeros((raw_actions.shape[0], 19)).astype(raw_actions.dtype)
                        actions[:, 8:16] = raw_actions
                        
                        action_queue.extend(actions[:NUM_STEPS])

                    raw_action = action_queue.popleft()
                    smoothed_action = ema.update(raw_action)

                    # ---- 键盘介入：检查是否有 XYZ delta ----
                    if keyboard_intervention is not None and keyboard_intervention.is_active:
                        delta_xyz, gripper_toggle = keyboard_intervention.get_delta()

                        # deploy_common.RealEnv.step：首次 IK 步切 wr_wbc_right_ik +
                        # send_action_right（TCP xyz + delta）；按 x 后非 IK 步切回 xr_wbc
                        obs = env.step(smoothed_action, ik_use=True, ee_delta=delta_xyz)

                        # IK 指令发送后，等待短暂时间让 whole_body_data 更新
                        time.sleep(0.01)

                        ik_solved_cmd = (
                            wb_cmd_reader.get_latest_cmd() if wb_cmd_reader is not None else None
                        )
                        processed = _merge_wholebody_ik_right_arm8(smoothed_action, ik_solved_cmd)
                        target_dim = int(np.asarray(smoothed_action, dtype=np.float32).ravel().shape[0])
                        row = _resize_action_vector(processed, target_dim)
                        ema.reset(row)
                        if ik_solved_cmd is not None and obs.get("state") is not None:
                            st = np.asarray(obs["state"], dtype=np.float32).ravel()
                            ic = np.asarray(ik_solved_cmd, dtype=np.float32).ravel()
                            if ic.shape == st.shape:
                                obs["state"] = ic.copy()

                        # 仅更新本步：用 IK 合并结果重置 EMA；action_queue 中后续步仍为原 VLA chunk，不清空不重填。
                        if ik_solved_cmd is None:
                            logging.debug(
                                "[Intervention] 尚未收到 whole_body_data，processed 退化为本步 smoothed_action"
                            )
                    else:
                        # 正常 policy 执行
                        obs = env.step(smoothed_action)

                    t += 1

            except KeyboardInterrupt:
                logging.info("Episode 被用户中断.")
                env.robot.cancel_servo_control()
            finally:
                if keyboard_intervention is not None:
                    keyboard_intervention.stop()

            episode_idx += 1
    finally:
        env.close()
        logging.info("环境已关闭.")
