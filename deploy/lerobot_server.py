import logging
import asyncio
from typing import Any
import numpy as np
import torch
import websockets
from websockets.server import serve
import openpi_client.msgpack_numpy as msgpack_numpy

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LeRobotPolicy:
    """Encapsulates the LeRobot policy inference logic (与 lerobot_deploy.LeRobotPolicy 对齐)."""

    def __init__(self, model_path: str):
        logging.info(f"Loading model from {model_path}...")
        cfg = PreTrainedConfig.from_pretrained(model_path)
        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(model_path)
        self.device = cfg.device or "cuda"
        self.policy.to(self.device)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=model_path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        self.policy.eval()
        logging.info("Model loaded successfully.")

    def infer(self, obs_dict: dict) -> np.ndarray:
        # 浅拷贝，避免 pop 破坏调用方缓存的 dict
        policy_obs_np = dict(obs_dict)
        task = policy_obs_np.pop("instruction")

        policy_obs_torch = prepare_observation_for_inference(
            policy_obs_np,
            device=self.device,
            task=task,
        )

        with torch.inference_mode():
            policy_obs_torch = self.preprocessor(policy_obs_torch)
            # select_action 通常只返回单步 (1, A)；开环需要整段 chunk 时用 predict_action_chunk（与 open_loop_eval / 训练一致）
            action_norm = self.policy.predict_action_chunk(policy_obs_torch)
            action = self.postprocessor(action_norm)

        action_np = action.detach().cpu().float().numpy()
        if action_np.ndim == 3 and action_np.shape[0] == 1:
            action_np = action_np[0]
        elif action_np.ndim == 2 and action_np.shape[0] == 1:
            action_np = action_np[0][None, :]
        elif action_np.ndim == 1:
            action_np = action_np[None, :]

        if action_np.ndim != 2:
            raise ValueError(f"Policy output action has unexpected shape: {action_np.shape}")

        return action_np.astype(np.float32, copy=False)


class LeRobotServer:
    def __init__(
        self,
        host="0.0.0.0",
        port=8000,
        model_path=None,
        *,
        image_keys: tuple[str, ...] = ("chest",),
    ):
        self.host = host
        self.port = port
        self.packer = msgpack_numpy.Packer()
        if model_path is None:
            raise ValueError("Model path must be provided")
        logging.info("Policy image keys (须与客户端一致): %s", image_keys)
        self.policy_node = LeRobotPolicy(model_path)

    async def handler(self, websocket):
        logging.info(f"Client connected: {websocket.remote_address}")
        
        # Send metadata (just like openpi server does, though our client might ignore it)
        metadata = {"name": "LeRobotServer", "version": "0.1"}
        await websocket.send(self.packer.pack(metadata))

        try:
            async for message in websocket:
                obs = msgpack_numpy.unpackb(message)
                try:
                    action = self.policy_node.infer(obs)
                    response = {"action": action}
                    await websocket.send(self.packer.pack(response))
                except Exception as e:
                    logging.error(f"Inference error: {e}")
                    await websocket.send(str(e)) # Send error string
        except websockets.ConnectionClosed:
            logging.info(f"Client disconnected: {websocket.remote_address}")

    async def serve(self):
        async with serve(self.handler, self.host, self.port, max_size=None, compression=None):
            logging.info(f"Server listening on {self.host}:{self.port}")
            await asyncio.Future()  # run forever

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/tars/code/lerobot/outputs/train/pi05_base_7.5k")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument(
        "--hand-test",
        action="store_true",
        help="与 build_lerobot_dataset --hand_test 一致：观测键 chest+scene_fisheye+left_wrist（须与客户端 lerobot_client_deploy 同参）。",
    )
    parser.add_argument(
        "--hand-test-no-chest",
        action="store_true",
        help="须与 --hand-test 同用：观测键 scene_fisheye、left_wrist、right_wrist。",
    )
    args = parser.parse_args()

    HAND_TEST = args.hand_test
    HAND_TEST_NO_CHEST = args.hand_test_no_chest
    if HAND_TEST_NO_CHEST and not HAND_TEST:
        parser.error("--hand-test-no-chest 必须与 --hand-test 同时使用")

    if HAND_TEST:
        image_keys = (
            ("scene_fisheye", "left_wrist", "right_wrist")
            if HAND_TEST_NO_CHEST
            else ("chest", "scene_fisheye", "left_wrist")
        )
    else:
        image_keys = ("chest",)

    server = LeRobotServer(
        host=args.host,
        port=args.port,
        model_path=args.model,
        image_keys=image_keys,
    )
    asyncio.run(server.serve())
