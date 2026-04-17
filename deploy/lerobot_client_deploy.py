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
from deploy_common import EMA, RolloutRecorder, RealEnv, Runner
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
        2. 逐步执行 action chunk；若介入激活且有键盘输入，
           将 XYZ delta 叠加到当前 action 并以 ik_use=True 发给 env.step()
        3. env.step() 发送 IK 指令后，从 whole_body/whole_body_data 读回 IK 解的
           joint_state_cmd，用该 cmd 覆盖当前这一步的 action
        4. 按下 x 键结束键盘控制，恢复 policy 自动执行
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
            logging.info(
                f"[Intervention] delta_xyz = [{self._delta[0]:+.4f}, "
                f"{self._delta[1]:+.4f}, {self._delta[2]:+.4f}]"
            )


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


class LeRobotClientPolicy:
    """Client for remote LeRobot policy inference."""

    def __init__(self, host: str, port: int, *, image_keys: Tuple[str, ...] = ("chest",)):
        self.client = WebsocketClientPolicy(host=host, port=port)
        self._image_keys = image_keys
        logging.info(f"Connected to LeRobot Policy Server at {host}:{port}")

    def __call__(self, obs: dict[str, Any]) -> np.ndarray:
        # 与 lerobot_deploy.LeRobotPolicy 一致的观测键，服务端用 prepare_observation_for_inference + instruction
        policy_obs: dict[str, Any] = {
            "observation.state": obs["state"],
            "instruction": obs["instruction"],
        }
        for k in self._image_keys:
            policy_obs[f"observation.images.{k}"] = obs[k]
        
        # infer() returns the dict unpacked from msgpack
        t0 = time.time()
        response = self.client.infer(policy_obs)
        latency_ms = (time.time() - t0) * 1000
        logging.info(f"Policy inference latency: {latency_ms:.2f} ms")
        
        # The server wraps the action in a dict: {"action": np.ndarray}
        if "action" not in response:
            raise ValueError(f"Unexpected response from server: {response.keys()}")
            
        action_np = response["action"]
        return action_np


def _validate_td3_metadata(
    meta: dict[str, Any],
    *,
    chunk_size: int,
    include_proprio: bool,
    warmup_steps: int,
    history_len: int,
    history_pad: str,
    task_description: str,
) -> None:
    """连接后与 serve_td3_policy.py 广播的 metadata 逐项对齐，避免两端 CLI 不一致导致静默错误。"""
    if meta.get("policy") != "td3_rlt":
        raise ValueError(
            f"服务端不是 TD3 策略（metadata.policy={meta.get('policy')!r}），"
            "请确认启动的是 serve_td3_policy.py"
        )
    checks: list[tuple[str, Any, Any]] = [
        ("chunk_size", meta.get("chunk_size"), chunk_size),
        ("include_proprio", meta.get("include_proprio"), include_proprio),
        ("warmup_steps", meta.get("warmup_steps"), warmup_steps),
        ("history_len", meta.get("history_len"), history_len),
        ("history_pad", meta.get("history_pad"), history_pad),
        ("task_description", meta.get("task_description"), task_description),
    ]
    bad = [(k, sv, cv) for k, sv, cv in checks if sv != cv]
    if bad:
        lines = [f"  {k}: 服务端={sv!r} 客户端={cv!r}" for k, sv, cv in bad]
        raise ValueError(
            "TD3 服务端与客户端配置不一致，请用相同参数启动 serve_td3_policy.py 与本脚本：\n"
            + "\n".join(lines)
        )
    logging.info("TD3 服务端 metadata 与客户端 CLI 已对齐: chunk_size=%s", chunk_size)


class TD3ClientPolicy:
    """TD3 策略客户端 — 发送完整关节状态（无预处理）给 serve_td3_policy.py 服务端，
    接收完整 env-space action chunk（含左臂前缀 / 腰部后缀），直接用于机器人执行。

    与 LeRobotClientPolicy 的关键区别：
      - 发送的是机器人原始全关节 state，不进行左/右臂拆分
      - 返回的 action 形状为 (chunk_size, full_action_dim)，无需客户端重建
      - 支持 episode 首帧 reset=True 通知服务端重置 state_builder 历史
      - 连接后校验服务端 metadata 与客户端 TD3 相关参数一致
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        image_keys: Tuple[str, ...] = ("chest",),
        chunk_size: int,
        include_proprio: bool,
        warmup_steps: int,
        history_len: int,
        history_pad: str,
        task_description: str,
    ) -> None:
        self.client = WebsocketClientPolicy(host=host, port=port)
        self._image_keys = image_keys
        meta = self.client.get_server_metadata()
        _validate_td3_metadata(
            meta,
            chunk_size=chunk_size,
            include_proprio=include_proprio,
            warmup_steps=warmup_steps,
            history_len=history_len,
            history_pad=history_pad,
            task_description=task_description,
        )
        logging.info(f"Connected to TD3 Policy Server at {host}:{port}")

    def infer(self, obs: dict[str, Any], *, reset: bool = False) -> np.ndarray:
        """发送完整 obs，返回 shape=(chunk_size, full_action_dim) 的 action chunk。

        Args:
            obs:   机器人原始观测字典，包含 "state"（全关节）和各路图像
            reset: True 时通知服务端重置 episode 内部状态（每 episode 首帧置 True）
        """
      # print(f"[TD3ClientPolicy.infer] obs.keys() = {list(obs.keys())}", flush=True)
      #  print(f"[TD3ClientPolicy.infer] self._image_keys = {self._image_keys}", flush=True)

        policy_obs: dict[str, Any] = {
            "observation.state": obs["state"],   # 完整关节状态，服务端负责拆分
            "instruction": obs.get("instruction", ""),
            "reset": reset,
        }
        for k in self._image_keys:
         #   print(f"[TD3ClientPolicy.infer] 添加图像键 '{k}'...", flush=True)
            policy_obs[f"observation.images.{k}"] = obs[k]

        t0 = time.time()

       # print("TD3 obs:", policy_obs,flush=True)

        response = self.client.infer(policy_obs)
        latency_ms = (time.time() - t0) * 1000
        logging.info(f"TD3 inference latency: {latency_ms:.2f} ms")

        if "action" not in response:
            raise ValueError(f"Unexpected response from TD3 server: {response.keys()}")
        
      #  print("TD3 action:", response["action"],flush=True)
        
        return response["action"]  # shape: (chunk_size, full_action_dim)

# ================================= 运行入口 =================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-robot", action="store_true")
    parser.add_argument("--allow-missing-images", action="store_true",
                        help="相机 topic 未收到数据时，用 640×480 全零 RGB 占位，避免卡住")
    parser.add_argument("--hand-no-basegripper", action="store_true",
                        help="29-dim action: 前12维发/hand_cmd，后17维由control处理(gripper/base=0)")
    parser.add_argument(
        "--hand-test",
        action="store_true",
        help="与 build_lerobot_dataset --hand_test 一致：观测含 chest+scene_fisheye+left_wrist，state/action 19 维含夹爪、无底盘；不可与 --hand-no-basegripper 同用。",
    )
    parser.add_argument(
        "--hand-test-no-chest",
        action="store_true",
        help="须与 --hand-test 同用：图像仅 scene_fisheye、left_wrist、right_wrist，无 chest（须与预训练 policy 的图像键一致）。",
    )
    parser.add_argument("--host", default="192.168.21.99")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument(
        "--num-open-loop-steps",
        type=int,
        default=50,
        help=(
            "每轮从服务端返回的 action chunk 中取前 N 步执行；"
            "--td3-mode 时建议与 serve_td3_policy.py 的 --chunk-size 一致"
        ),
    )
    parser.add_argument("--task-description", type=str,
                        default="pick up the box on the table and stack it on the box on the floor",
                        help="任务描述字符串，发送给服务端")
    # ---- TD3 模式 ----
    parser.add_argument(
        "--td3-mode",
        action="store_true",
        help="连接 serve_td3_policy.py 服务端：发送完整关节状态，接收完整 env-space action chunk（无需客户端前后处理）",
    )
    parser.add_argument("--ema-alpha", type=float, default=1.0,
                        help="TD3 模式下动作 EMA 平滑系数，1.0 = 不平滑")
    parser.add_argument(
        "--keyboard-ee-intervention",
        action="store_true",
        help="启用键盘末端介入：SPACE 开始介入，w/s/a/d/q/e XYZ 平移，g 夹爪，+/- 步长，x 结束介入恢复 policy，ESC 退出 episode",
    )
    # 以下与 serve_td3_policy.py / train_td3_online_cli.py 对齐；--td3-mode 时连接后与服务端 metadata 校验
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="须与 serve_td3_policy.py 一致；TD3 模式连接时与服务端校验",
    )
    parser.add_argument(
        "--include-proprio",
        action="store_true",
        help="须与 serve_td3_policy.py 一致（TD3 模式）",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="须与 serve_td3_policy.py 一致（TD3 模式，仅用于校验 metadata）",
    )
    parser.add_argument("--history-len", type=int, default=1, help="须与 serve_td3_policy.py 一致（TD3 模式）")
    parser.add_argument(
        "--history-pad",
        type=str,
        choices=("zero", "repeat"),
        default="repeat",
        help="须与 serve_td3_policy.py 一致（TD3 模式）",
    )
    args = parser.parse_args()

    HOST = args.host
    PORT = args.port
    NUM_STEPS = args.num_open_loop_steps
    INIT_ROBOT = args.init_robot
    HAND_NO_BASEGRIPPER = args.hand_no_basegripper
    HAND_TEST = args.hand_test
    HAND_TEST_NO_CHEST = args.hand_test_no_chest
    TASK_DESCRIPTION = args.task_description

    if HAND_TEST and HAND_NO_BASEGRIPPER:
        parser.error("--hand-test 与 --hand-no-basegripper 不能同时使用")
    if HAND_TEST_NO_CHEST and not HAND_TEST:
        parser.error("--hand-test-no-chest 必须与 --hand-test 同时使用")
    if args.td3_mode and HAND_NO_BASEGRIPPER:
        parser.error("--td3-mode 与 --hand-no-basegripper 不能同时使用")
    if args.td3_mode and args.num_open_loop_steps > args.chunk_size:
        parser.error(
            f"--num-open-loop-steps ({args.num_open_loop_steps}) 不能大于 --chunk-size ({args.chunk_size})"
        )

    if HAND_TEST:
        image_keys = (
            ("scene_fisheye", "left_wrist", "right_wrist")
            if HAND_TEST_NO_CHEST
            else ("chest", "scene_fisheye", "left_wrist")
        )
    else:
        image_keys = ("chest",)

    # TD3 模式默认使用 4 路相机（与 openpi insert_0402d 训练配置一致）。
    # 若 RealEnv 实际订阅的相机路数不同，需与此保持一致（同时调整 deploy_common.py 里的 rgb_topics）。
    if args.td3_mode and not HAND_TEST:
        image_keys = ("chest", "scene_fisheye", "left_wrist", "right_wrist")
    # ======================== TD3 部署模式 ========================
    # 与 serve_td3_policy.py 配套使用：
    #   - 客户端发送全关节原始 obs（不做左/右臂拆分）
    #   - 服务端完成 VLA 编码 + TD3 推理 + prefix/suffix 重建
    #   - 客户端接收完整 action chunk，直接步进机器人
    if args.td3_mode:
        policy = TD3ClientPolicy(
            host=HOST,
            port=PORT,
            image_keys=image_keys,
            chunk_size=args.chunk_size,
            include_proprio=args.include_proprio,
            warmup_steps=args.warmup_steps,
            history_len=args.history_len,
            history_pad=args.history_pad,
            task_description=TASK_DESCRIPTION,
        )

        print(f"start_policy")

        env = RealEnv(
            init_robot=INIT_ROBOT,
            hand_test=HAND_TEST,
            hand_test_no_chest=HAND_TEST_NO_CHEST,
            allow_missing_images=args.allow_missing_images,
            td3_mode=True,
        )

        print(f"start_env")

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
                
                print(f"obs-get")
                obs = env.reset_td3()      # TD3 模式：直接取观测，不触发伺服使能等待
                action_queue: deque = deque()
                ema = EMA(alpha=args.ema_alpha)
                first_call = True
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
                            actions = policy.infer(obs, reset=first_call)
                            first_call = False
                            action_queue.extend(actions[:NUM_STEPS])

                        raw_action = action_queue.popleft()
                        smoothed_action = ema.update(raw_action)

                        # ---- 键盘介入：检查是否有 XYZ delta ----
                        if keyboard_intervention is not None and keyboard_intervention.is_active and keyboard_intervention.has_pending_delta():
                            delta_xyz, gripper_toggle = keyboard_intervention.get_delta()
                            logging.info(
                                f"[Intervention] 发送 EE delta: "
                                f"[{delta_xyz[0]:+.4f}, {delta_xyz[1]:+.4f}, {delta_xyz[2]:+.4f}], "
                                f"gripper_toggle={gripper_toggle}"
                            )

                            # TODO: 在这里实现将 delta_xyz 转换为 IK 指令并通过 env.step 发送
                            # ========================================================
                            # 你需要实现 env.step(smoothed_action, ik_use=True, ee_delta=delta_xyz)
                            # 该方法内部应：
                            #   1. 基于当前末端位姿 + delta_xyz 计算目标笛卡尔位姿
                            #   2. 通过 PoseTrackingTarget 发送给 xr_wbc 做 IK 解算
                            #   3. 返回新的 obs
                            # ========================================================
                            obs = env.step(smoothed_action, ik_use=True, ee_delta=delta_xyz)

                            # IK 指令发送后，等待短暂时间让 whole_body_data 更新
                            time.sleep(0.01)

                            # 读取 IK 解算后的 joint cmd，覆盖本步 action（仅当前这一步）
                            if wb_cmd_reader is not None:
                                ik_solved_cmd = wb_cmd_reader.get_latest_cmd()
                                if ik_solved_cmd is not None:
                                    # 用 IK 解的 joint cmd 覆盖 ema 状态，使后续步平滑过渡
                                    ema.reset(ik_solved_cmd)
                                    logging.info("[Intervention] IK cmd 已覆盖当前步 action")
                                else:
                                    logging.warning("[Intervention] 尚未收到 whole_body_data，无法覆盖 action")
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

    # ======================== 原始 LeRobot 部署模式 ========================
    else:
        policy = LeRobotClientPolicy(host=HOST, port=PORT, image_keys=image_keys)

        runner = Runner(
            policy=policy,
            num_open_loop_steps=NUM_STEPS,
            ema_alpha=1.0,
            records=["image", "action", "state"],
            init_robot=INIT_ROBOT,
            hand_no_basegripper=HAND_NO_BASEGRIPPER,
            hand_test=HAND_TEST,
            hand_test_no_chest=HAND_TEST_NO_CHEST,
            allow_missing_images=args.allow_missing_images,
        )

        episode_idx = 0

        try:
            while True:
                cmd = input("Press Enter to start new episode, or type 'q' to quit... ")
                if cmd.strip().lower() in {"q", "quit", "exit"}:
                    break
                final_obs = runner.run_episode(
                    idx=episode_idx,
                    task_description=TASK_DESCRIPTION,
                    max_steps=50000
                )
                episode_idx += 1
        finally:
            runner.env.close()
            logging.info("环境已关闭.")
