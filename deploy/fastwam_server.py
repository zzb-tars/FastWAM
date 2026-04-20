"""
FastWAM WebSocket Server for Remote Policy Inference

Usage:
    # From FastWAM directory (mock server for testing)
    conda run -n fastwam_libero_v0 python deploy/mock_server.py

    # Real model server
    FASTWAM_PORT=8002 conda run -n fastwam_libero_v0 python deploy/fastwam_server.py

    # Override checkpoint
    conda run -n fastwam_libero_v0 python deploy/fastwam_server.py \
        inference.checkpoint_path=/path/to/checkpoint.pt \
        inference.device=cuda:0

    # Stop: Ctrl+C
"""

import asyncio
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
import torchvision
import websockets
from omegaconf import DictConfig

# Setup path before any imports
# deploy/fastwam_server.py -> parents[1] is FastWAM root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)
register_default_resolvers()

# Default prompt template (from robot_video_dataset.py)
DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

# Force stdout flush and file logging for visibility
_LOG_FILE = None

def _init_file_logging():
    """Initialize file logging to capture output that stdout can't."""
    global _LOG_FILE
    log_file = "/tmp/fastwam_server.log"
    try:
        _LOG_FILE = open(log_file, "a", buffering=1)  # Line buffering
        _log(f"[FILE_LOG_INIT] Logging to {log_file}")
    except Exception as e:
        print(f"Failed to init file logging: {e}", file=sys.stderr, flush=True)

def _log(msg: str):
    """Print to both stdout and file."""
    try:
        print(msg, file=sys.stdout, flush=True)
    except:
        pass
    
    if _LOG_FILE is not None:
        try:
            _LOG_FILE.write(msg + "\n")
            _LOG_FILE.flush()
        except:
            pass


# ── numpy-safe msgpack helpers (compatible with client msgpack_numpy) ───────
# Import msgpack_numpy from openpi_client if available, otherwise use fallback
try:
    # Try to use the same msgpack_numpy as the client
    from openpi_client import msgpack_numpy as client_msgpack_numpy
    _pack = client_msgpack_numpy.packb
    _unpack = client_msgpack_numpy.unpackb
except ImportError:
    # Fallback: implement compatible msgpack_numpy encoding locally
    import msgpack
    import functools
    
    def _pack_array(obj):
        """Encode numpy arrays compatible with msgpack_numpy library."""
        if (isinstance(obj, (np.ndarray, np.generic)) and 
            obj.dtype.kind in ("V", "O", "c")):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        
        if isinstance(obj, np.ndarray):
            return {
                b"__ndarray__": True,
                b"data": obj.tobytes(),
                b"dtype": obj.dtype.str,
                b"shape": obj.shape,
            }
        
        if isinstance(obj, np.generic):
            return {
                b"__npgeneric__": True,
                b"data": obj.item(),
                b"dtype": obj.dtype.str,
            }
        
        return obj
    
    def _unpack_array(obj):
        """Decode numpy arrays from msgpack."""
        if isinstance(obj, dict):
            if b"__ndarray__" in obj:
                return np.ndarray(
                    buffer=obj[b"data"],
                    dtype=np.dtype(obj[b"dtype"]),
                    shape=obj[b"shape"]
                )
            if b"__npgeneric__" in obj:
                return np.dtype(obj[b"dtype"]).type(obj[b"data"])
        return obj
    
    _Packer = functools.partial(msgpack.Packer, default=_pack_array)
    _pack_func = functools.partial(msgpack.packb, default=_pack_array)
    _unpack_func = functools.partial(msgpack.unpackb, object_hook=_unpack_array)
    
    def _pack(data) -> bytes:
        return _pack_func(data, use_bin_type=True)
    
    def _unpack(data):
        return _unpack_func(data, raw=False)


class FastWAMPolicy:
    """Encapsulates FastWAM model inference logic."""

    def __init__(self, cfg: DictConfig):
        """Initialize FastWAM model from Hydra config.
        
        Args:
            cfg: Hydra DictConfig with model/data/inference settings
        """
        infer_cfg = cfg.inference
        self.device = infer_cfg.get("device", "cuda:0")
        self.torch_dtype = self._resolve_dtype(cfg.get("mixed_precision", "no"))
        
        # Text embedding cache config - expand path properly
        text_cache_raw = infer_cfg.get("text_embedding_cache_dir", None)
        if text_cache_raw is not None:
            # Expand user home and environment variables first
            text_cache_expanded = os.path.expanduser(os.path.expandvars(text_cache_raw))
            # Convert to absolute path (relative paths are relative to project root)
            if not os.path.isabs(text_cache_expanded):
                text_cache_expanded = os.path.join(project_root, text_cache_expanded)
            self.text_embedding_cache_dir = text_cache_expanded
        else:
            self.text_embedding_cache_dir = None
        
        self.context_len = infer_cfg.get("context_len", 128)
        
        # Load model
        _log(f"[INFO] Loading FastWAM model...")
        from hydra.utils import instantiate
        self.model = instantiate(
            cfg.model,
            model_dtype=self.torch_dtype,
            device=self.device,
        )
        
        # Load checkpoint from config
        checkpoint_path = infer_cfg.get("checkpoint_path", None)
        if checkpoint_path is not None:
            ckpt_path = Path(os.path.expanduser(os.path.expandvars(checkpoint_path)))
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            _log(f"[INFO] Loading checkpoint: {ckpt_path}")
            self.model.load_checkpoint(str(ckpt_path))
        else:
            _log(f"[WARNING] No checkpoint_path in config; using base weights")
        
        self.model.eval()
        _log(f"[INFO] FastWAM model loaded successfully")
        
        # Load processor for denormalization
        self.processor = None
        dataset_stats_path = infer_cfg.get("dataset_stats_path", None)
        if dataset_stats_path is not None:
            self._load_denormalization_processor(cfg, dataset_stats_path)
        else:
            _log(f"[WARNING] No dataset_stats_path in config; denormalization disabled")
        
        # Store inference config (all from config, not from runtime overrides)
        self.action_horizon = infer_cfg.get("action_horizon", 64)
        self.num_inference_steps = infer_cfg.get("num_inference_steps", 5)
        self.num_video_frames = infer_cfg.get("num_video_frames", 1)
        self.seed = infer_cfg.get("seed", 42)
        self.tiled = infer_cfg.get("tiled", False)
        
        _log(f"[INFO] Config: action_horizon={self.action_horizon}, num_inference_steps={self.num_inference_steps}")
        if self.text_embedding_cache_dir:
            _log(f"[INFO] Text embedding cache: {self.text_embedding_cache_dir}")
            if not os.path.exists(self.text_embedding_cache_dir):
                _log(f"[WARNING] Cache directory not found: {self.text_embedding_cache_dir}")
        else:
            _log(f"[INFO] Text embedding cache: disabled (using random context)")

    def _load_denormalization_processor(self, cfg: DictConfig, dataset_stats_path: str):
        """Load processor and normalizer from dataset_stats for action denormalization."""
        try:
            from hydra.utils import instantiate
            
            stats_path = Path(os.path.expanduser(os.path.expandvars(dataset_stats_path)))
            if not stats_path.exists():
                _log(f"[WARNING] Dataset stats not found: {stats_path}")
                return
            
            _log(f"[INFO] Loading dataset processor from: {stats_path}")
            
            # Load dataset with pretrained normalization stats
            dataset = instantiate(
                cfg.data.train,
                pretrained_norm_stats=str(stats_path)
            )
            self.processor = dataset.lerobot_dataset.processor
            self.normalizer = self.processor.normalizer
            self.action_state_merger = self.processor.action_state_merger
            self.shape_meta = self.processor.shape_meta
            _log(f"[INFO] Denormalization processor loaded successfully")
        except Exception as e:
            _log(f"[WARNING] Failed to load processor: {e}")
            self.processor = None

    @staticmethod
    def _resolve_dtype(mixed_precision: str) -> torch.dtype:
        """Convert mixed_precision string to torch dtype."""
        mp = str(mixed_precision).strip().lower()
        if mp == "fp16":
            return torch.float16
        elif mp == "bf16":
            return torch.bfloat16
        else:
            return torch.float32

    def _get_cached_text_context(self, prompt: str):
        """Load cached text embedding for the prompt.
        
        Args:
            prompt: Task instruction text
            
        Returns:
            (context, context_mask): Cached embeddings from disk
        """
        if self.text_embedding_cache_dir is None:
            raise ValueError(
                "text_embedding_cache_dir is not set. "
                "Please set inference.text_embedding_cache_dir in config."
            )
        
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Hash the prompt to get cache filename
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(
            cache_dir, 
            f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )
        
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                f" prompt='{prompt}' "
                "Run scripts/precompute_text_embeds.py first."
            )
        
        # Load cached embeddings
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        
        # Validate shapes
        if context.ndim != 2:
            raise ValueError(
                f"Cached context must be 2D [L, D], got {tuple(context.shape)}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached mask must be 1D [L], got {tuple(context_mask.shape)}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Context length mismatch: expected {self.context_len}, got {context.shape[0]}"
            )
        
        return context, context_mask

    def _denormalize_action(
        self,
        action: torch.Tensor,
        proprio: torch.Tensor,
    ) -> np.ndarray:
        """Denormalize predicted action from model output space to real action space.
        
        This mirrors the logic in experiments/infer/infer_action_offline.py.
        
        Args:
            action: [T_action, action_dim] normalized action from model
            proprio: [T_action, state_dim] proprioception
            
        Returns:
            [T_action, action_dim] raw (denormalized) action
        """
        if not self.processor:
            logger.warning("No processor available; returning action as-is")
            return action.detach().cpu().numpy().astype(np.float32)
        
        if action.ndim == 2:
            action = action.unsqueeze(0)  # [1, T, D]
        if action.ndim != 3:
            raise ValueError(f"Expected action ndim=3, got {action.ndim}")
        
        action = action.detach().to(device="cpu", dtype=torch.float32)
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(0)  # [1, T, D]
        elif proprio.ndim == 1:
            # Repeat single state across action horizon
            action_horizon = action.shape[1]
            proprio = proprio.unsqueeze(0).repeat(1, action_horizon, 1)  # [1, T, D]
        
        proprio = proprio.detach().to(device="cpu", dtype=torch.float32)
        
        action_meta = self.shape_meta["action"]
        state_meta = self.shape_meta["state"]
        
        # Denormalization pipeline: merger.backward -> normalizer.backward -> merger.forward
        batch = {"action": action, "state": proprio}
        batch = self.action_state_merger.backward(batch)
        batch = self.normalizer.backward(batch)
        
        merged_batch = {
            "action": {m["key"]: batch["action"][m["key"]].squeeze(0) for m in action_meta},
            "state": {m["key"]: batch["state"][m["key"]].squeeze(0) for m in state_meta},
        }
        merged_batch = self.action_state_merger.forward(merged_batch)
        denorm_action = merged_batch["action"].unsqueeze(0)
        return denorm_action.numpy()  # [1, T, D]

    def _image_transform(self, image_np: np.ndarray) -> torch.Tensor:
        image_tensor = torch.from_numpy(image_np).to(
            device=self.device,
            dtype=torch.float32,
        )
        image_tensor = image_tensor / 255.0
        resize_func = torchvision.transforms.Resize(
            size=(224, 224), 
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR, 
            max_size=None, antialias=True)
        image_tensor = resize_func(image_tensor)
        return image_tensor

    def _stack_image(self, image_tensor_list) -> torch.Tensor:
        image = torch.stack(image_tensor_list, dim=0)  # [N, C, H, W]
        image = torch.cat([image[i] for i in range(len(image_tensor_list))], dim=-1)  # [C, H, W*N]
        return image
    
    def _normalize_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        import torchvision.transforms.functional as transforms_F
        image_tensor = transforms_F.normalize(
            image_tensor,
            mean=0.5,
            std=0.5,
        )
        return image_tensor

    def infer(self, obs_dict: dict) -> np.ndarray:
        """Run inference on observation.
        
        All config (action_horizon, num_inference_steps, etc.) comes from Hydra config,
        not from client obs_dict. State is already normalized (from LeRobot dataset).
        
        Args:
            obs_dict: Dictionary containing:
                - "observation.image.chest": np.ndarray of shape [3, H, W] (uint8 in [0, 255])
                - "observation.image.right_wrist": np.ndarray of shape [3, H, W] (uint8 in [0, 255])
                - "state": np.ndarray of shape [state_dim] (already normalized)
                - "instruction": str (optional, task description)
                
        Returns:
            np.ndarray of shape [action_horizon, action_dim] (denormalized)
        """
        ######### Extract image from observation
        image_chest_np = obs_dict.get("observation.image.chest")
        image_right_wrist_np = obs_dict.get("observation.image.right_wrist")
        if image_chest_np is None or image_right_wrist_np is None:
            raise ValueError("obs_dict must contain 'observation.image.chest' and 'observation.image.right_wrist'")
        if image_chest_np.dtype != np.uint8 or image_right_wrist_np.dtype != np.uint8:
            raise ValueError("Expected image dtype uint8, got {}".format(image_chest_np.dtype))
        if image_chest_np.shape[0] != 3 or image_chest_np.shape[1] != 480 or image_chest_np.shape[2] != 640:
            raise ValueError("Expected image shape [3, 480, 640], got {}".format(image_chest_np.shape))
        if image_right_wrist_np.shape[0] != 3 or image_right_wrist_np.shape[1] != 480 or image_right_wrist_np.shape[2] != 640:
            raise ValueError("Expected image shape [3, 480, 640], got {}".format(image_right_wrist_np.shape))
        image_chest_tensor = self._image_transform(image_chest_np)
        image_right_wrist_tensor = self._image_transform(image_right_wrist_np)
        image_tensor = self._stack_image([image_chest_tensor, image_right_wrist_tensor])
        image_tensor = self._normalize_image(image_tensor)
        image_tensor = image_tensor.to(device=self.device, dtype=self.torch_dtype)
        if image_tensor.ndim != 3 or image_tensor.shape[0] != 3 or image_tensor.shape[1] != 224 or image_tensor.shape[2] != 448:
            raise ValueError(f"Expected image tensor shape [3, 224, 448], got {tuple(image_tensor.shape)}")

        
        ######### Proprioception (state)
        state_np = obs_dict.get("state", None)
        # Create on CPU so normalizer (usually on CPU) doesn't cause device mismatch
        state_tensor = torch.from_numpy(state_np).to(
            device="cpu",
            dtype=torch.float32,
        )
        if state_tensor.ndim != 2 or state_tensor.shape[0] != 1 or state_tensor.shape[1] != 8:
            raise ValueError(f"Expected state shape [1, state_dim], got {tuple(state_tensor.shape)}")
        state_tensor = self.normalizer.normalizers["state"]["default"].forward(state_tensor)
        proprio_tensor = state_tensor.to(device=self.device, dtype=self.torch_dtype)
        
        ######### Format prompt and load cached text embedding
        instruction = obs_dict.get("instruction", "")
        prompt = DEFAULT_PROMPT.format(task=instruction)
        try:
            context, context_mask = self._get_cached_text_context(prompt)
        except Exception as e:
            _log(f"[WARNING] Failed to load cached context: {e}")
            # Fallback: use dummy context if cache is not available
            batch_size = 1
            context_dim = 4096
            seq_len = self.context_len
            context = torch.randn(seq_len, context_dim, dtype=self.torch_dtype)
            context_mask = torch.ones(seq_len, dtype=torch.bool)
        # Move context to device
        context = context.to(device=self.device, dtype=self.torch_dtype).unsqueeze(0)
        context_mask = context_mask.to(device=self.device).unsqueeze(0)
        # Apply mask (consistent with robot_video_dataset.py)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)  # All True mask after masking
        
        # Run inference with config parameters
        with torch.no_grad():
            infer_kwargs = {
                "input_image": image_tensor,
                "action_horizon": self.action_horizon,
                "proprio": proprio_tensor,
                "num_inference_steps": self.num_inference_steps,
                "seed": self.seed,
                "tiled": self.tiled,
                "context": context,
                "context_mask": context_mask,
            }

            output = self.model.infer_action(prompt=None, **infer_kwargs)
        
        # Extract normalized action from model output
        action_tensor = output["action"]  # [action_horizon, action_dim]
        action_normalized = action_tensor.detach().cpu().numpy().astype(np.float32)
        
        # Denormalize action if processor is available
        action_denormalized = None
        if self.processor is not None and proprio_tensor is not None:
            try:
                action_denormalized_raw = self._denormalize_action(action_tensor, proprio_tensor)
                action_denormalized = action_denormalized_raw.squeeze(0)  # [T, D]
            except Exception as e:
                _log(f"[WARNING] Denormalization failed: {e}; using normalized action")
                action_denormalized = action_normalized
        else:
            action_denormalized = action_normalized
        
        return action_denormalized


class FastWAMServer:
    """WebSocket server for FastWAM policy inference."""

    def __init__(
        self,
        cfg: DictConfig,
        host: str = "0.0.0.0",
        port: int = 7880,
    ):
        """Initialize server from Hydra config.
        
        Args:
            cfg: Hydra configuration
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        
        _log(f"[INFO] Initializing FastWAM policy from config...")
        self.policy = FastWAMPolicy(cfg)
        
        _log(f"[INFO] Server initialized: {host}:{port}")

    async def handler(self, websocket):
        """Handle client connection."""
        client_addr = websocket.remote_address
        _log(f"[INFO] Client connected: {client_addr}")
        
        # Send server metadata
        metadata = {
            "name": "FastWAMServer",
            "version": "1.0",
            "model": "FastWAM",
        }
        try:
            await websocket.send(_pack(metadata))
            _log(f"[DEBUG] Metadata sent")
        except Exception as e:
            _log(f"[ERROR] Error sending metadata: {e}")
            return
        
        # Main inference loop
        _log(f"[DEBUG] Entering message receive loop...")
        message_count = 0
        try:
            async for message in websocket:
                message_count += 1
                # _log(f"[DEBUG] Message #{message_count} received: {len(message)} bytes")
                try:
                    # Unpack observation
                    obs = _unpack(message)
                    _log(f"[INFO] Received obs keys={list(obs.keys())}")
                    
                    # Run inference
                    import time as _time
                    _t0 = _time.perf_counter()
                    _log(f"[DEBUG] Starting inference...")
                    try:
                        action = self.policy.infer(obs)
                        _dt = _time.perf_counter() - _t0
                        _log(f"[DEBUG] Inference completed in {_dt*1000:.1f} ms, action shape: {action.shape}")
                    except Exception as infer_err:
                        _dt = _time.perf_counter() - _t0
                        _log(f"[ERROR] Inference failed after {_dt*1000:.1f} ms: {infer_err}")
                        import traceback
                        _log(f"[ERROR] Traceback:\n{traceback.format_exc()}")
                        raise
                    
                    # Pack response
                    response = {"action": action}
                    packed = _pack(response)
                    await websocket.send(packed)
                    _log(f"[INFO] Sent action shape={action.shape} ({_dt*1000:.1f} ms)")
                    
                except Exception as e:
                    import traceback
                    _log(f"[ERROR] Inference error: {e}")
                    _log(f"[ERROR] Traceback:\n{traceback.format_exc()}")
                    try:
                        await websocket.send(_pack({"error": str(e)}))
                    except Exception as send_err:
                        _log(f"[ERROR] Failed to send error response: {send_err}")
                        
        except websockets.ConnectionClosed:
            _log(f"[INFO] Client disconnected: {client_addr}")
        except Exception as e:
            import traceback
            _log(f"[ERROR] Handler error: {e}")
            _log(f"[ERROR] Traceback:\n{traceback.format_exc()}")

    async def serve(self):
        """Start server."""
        async with websockets.serve(
            self.handler,
            self.host,
            self.port,
            ping_interval=None,    # Disable ping/pong (inference is long-running)
            max_size=None,
            compression=None,
        ):
            _log(f"[INFO] FastWAM Server listening on ws://{self.host}:{self.port}")
            _log(f"[INFO] Press Ctrl+C to stop.")
            await asyncio.Future()  # Run forever


@hydra.main(version_base="1.3", config_path=str(project_root / "configs"), config_name="infer_x1_insert")
def main(cfg: DictConfig):
    """Main entry point - all config from Hydra, no CLI overrides needed."""
    # Initialize file logging
    _init_file_logging()
    
    # Read server config
    infer_cfg = cfg.inference
    host = infer_cfg.get("server_host", "0.0.0.0")
    port = infer_cfg.get("server_port", 7880)

    _log(f"[INFO] Server configuration:")
    _log(f"[INFO]   host              = {host}")
    _log(f"[INFO]   port              = {port}")
    _log(f"[INFO]   device            = {infer_cfg.get('device', 'cuda:0')}")
    _log(f"[INFO]   checkpoint        = {infer_cfg.get('checkpoint_path', 'N/A')}")
    _log(f"[INFO]   action_horizon    = {infer_cfg.get('action_horizon', 16)}")
    _log(f"[INFO]   num_infer_steps   = {infer_cfg.get('num_inference_steps', 20)}")
    _log(f"[INFO]   dataset_stats     = {infer_cfg.get('dataset_stats_path', 'N/A')}")
    cache_dir_raw = infer_cfg.get('text_embedding_cache_dir', None)
    if cache_dir_raw:
        cache_dir_expanded = os.path.expanduser(os.path.expandvars(cache_dir_raw))
        if not os.path.isabs(cache_dir_expanded):
            cache_dir_expanded = os.path.join(project_root, cache_dir_expanded)
        _log(f"[INFO]   text_embed_cache  = {cache_dir_expanded}")
    else:
        _log(f"[INFO]   text_embed_cache  = disabled")
    _log(f"[INFO]   context_len       = {infer_cfg.get('context_len', 128)}")

    srv = FastWAMServer(cfg, host=host, port=int(port))
    asyncio.run(srv.serve())


if __name__ == "__main__":
    main()
