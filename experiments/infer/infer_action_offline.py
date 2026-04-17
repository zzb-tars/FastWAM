"""
Offline inference script for FastWAM on custom datasets.

Loads a trained checkpoint and runs action-only inference (infer_action) on
samples drawn from the training dataset. Saves predicted actions and
optionally input/GT videos for visual inspection.

Usage:
    python experiments/infer/infer_action_offline.py --config-name infer_x1_insert
    python experiments/infer/infer_action_offline.py --config-name infer_x1_insert \
        inference.checkpoint_path=/path/to/other/checkpoint.pt \
        inference.num_samples=20
"""
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import hydra
import matplotlib
import numpy as np
import torch
import time
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging
from fastwam.utils.video_io import save_mp4
from fastwam.utils import misc

logger = get_logger(__name__)

register_default_resolvers()


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(f"Unsupported mixed_precision: {mixed_precision}")
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _load_model(cfg: DictConfig, device: str):
    """Instantiate FastWAM model and load trained checkpoint."""
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)

    ckpt_path = cfg.inference.checkpoint_path
    if ckpt_path is not None:
        ckpt = Path(os.path.expanduser(os.path.expandvars(str(ckpt_path))))
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        logger.info("Loading checkpoint: %s", ckpt)
        model.load_checkpoint(str(ckpt))
    else:
        logger.warning("No checkpoint_path specified; using base pretrained weights.")

    model.eval()
    return model


def _load_dataset(cfg: DictConfig):
    """Build the dataset with normalizer initialized from saved stats."""
    data_cfg = cfg.data
    stats_path = cfg.inference.dataset_stats_path
    if stats_path is None:
        raise ValueError("inference.dataset_stats_path must be set.")
    stats_path = Path(os.path.expanduser(os.path.expandvars(str(stats_path))))
    if not stats_path.exists():
        raise FileNotFoundError(f"Dataset stats not found: {stats_path}")

    # Register work_dir so dataset constructor can save stats copy
    output_dir = str(cfg.inference.output_dir)
    misc.register_work_dir(output_dir)

    # Instantiate dataset with pretrained normalization stats
    dataset = instantiate(data_cfg.train, pretrained_norm_stats=str(stats_path))
    return dataset


def _denormalize_action(
    action: torch.Tensor,
    proprio: torch.Tensor,
    processor,
) -> np.ndarray:
    """Denormalize predicted action from model output space back to raw action space.

    Follows the same logic as Wan22Trainer.evaluate():
        merger.backward -> normalizer.backward -> squeeze(0) -> merger.forward
    """
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"action must be [B, T, D] or [T, D], got shape {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    state_meta = processor.shape_meta["state"]

    action = action.detach().to(device="cpu", dtype=torch.float32)

    # proprio is [T_action, state_dim], we need [1, T_action, state_dim] for the batch
    if proprio.ndim == 2:
        proprio = proprio.unsqueeze(0)
    proprio = proprio.detach().to(device="cpu", dtype=torch.float32)

    batch = {"action": action, "state": proprio}
    batch = processor.action_state_merger.backward(batch)
    batch = processor.normalizer.backward(batch)
    # squeeze(0) to go from [1, T, D] -> [T, D] for merger.forward
    merged_batch = {
        "action": {m["key"]: batch["action"][m["key"]].squeeze(0) for m in action_meta},
        "state": {m["key"]: batch["state"][m["key"]].squeeze(0) for m in state_meta},
    }
    merged_batch = processor.action_state_merger.forward(merged_batch)
    denorm_action = merged_batch["action"].unsqueeze(0)  # [1, T, D]
    return denorm_action.numpy()


def _video_tensor_to_pil_frames(video: torch.Tensor) -> list:
    """Convert [C, T, H, W] tensor in [-1, 1] to list of PIL Images."""
    video = video.detach().float().cpu().clamp(-1, 1)
    video = ((video + 1.0) * 127.5).to(torch.uint8)
    frames = []
    for t in range(video.shape[1]):
        frame = video[:, t].permute(1, 2, 0).numpy()  # [H, W, C]
        frames.append(Image.fromarray(frame))
    return frames


def _stitch_frames_vert(top_frames: list[Image.Image], bottom_frames: list[Image.Image]) -> list[Image.Image]:
    """Stitch two frame lists vertically frame-by-frame (top=GT, bottom=pred).
    
    When frame counts differ, pad the shorter sequence with its last frame.
    """
    if len(top_frames) == 0 or len(bottom_frames) == 0:
        return []
    
    num_frames = max(len(top_frames), len(bottom_frames))
    stitched = []
    for i in range(num_frames):
        # Use last frame as padding if one sequence is shorter
        top = top_frames[min(i, len(top_frames) - 1)].convert("RGB")
        bottom = bottom_frames[min(i, len(bottom_frames) - 1)].convert("RGB")
        if top.size != bottom.size:
            bottom = bottom.resize(top.size, resample=Image.BILINEAR)
        canvas = Image.new("RGB", (top.width, top.height + bottom.height))
        canvas.paste(top, (0, 0))
        canvas.paste(bottom, (0, top.height))
        stitched.append(canvas)
    return stitched


def _compute_action_error_metrics(pred_action_raw: np.ndarray, gt_action_raw: np.ndarray) -> dict:
    """Compute per-dimension and global action error metrics.

    NOTE: inputs must be DENORMALIZED (raw physical) actions, after calling _denormalize_action().
    For x1 robot:
      - dims 0-6 (delta_action_dim_mask=True):  per-step joint angle delta, unit = radians
      - dim  7   (delta_action_dim_mask=False): gripper absolute position, unit = [0, 100] scale
    MAE for joint dims can be interpreted as radians directly (e.g. 0.03 rad ≈ 1.7 deg).
    """
    if pred_action_raw.shape != gt_action_raw.shape:
        raise ValueError(
            "Action shape mismatch for metric computation: "
            f"pred={pred_action_raw.shape} vs gt={gt_action_raw.shape}"
        )
    diff = pred_action_raw - gt_action_raw
    abs_err = np.abs(diff)
    sq_err = diff ** 2
    mae_per_dim = abs_err.mean(axis=0)
    rmse_per_dim = np.sqrt(sq_err.mean(axis=0))
    return {
        "mae_global": float(abs_err.mean()),
        "rmse_global": float(np.sqrt(sq_err.mean())),
        "max_abs_error": float(abs_err.max()),
        "num_steps": int(abs_err.shape[0]),
        "mae_per_dim": [float(v) for v in mae_per_dim.tolist()],
        "rmse_per_dim": [float(v) for v in rmse_per_dim.tolist()],
        "unit": "raw (denormalized) action space: dims 0-6 in radians (delta), dim 7 gripper scaled to [0, 1] for this script",
    }


def _build_action_error_records(
    pred_action_raw: np.ndarray,
    gt_action_raw: np.ndarray,
    sample_idx: int,
) -> list[dict]:
    """Build per-sample, per-time-step, per-dimension absolute error records."""
    if pred_action_raw.shape != gt_action_raw.shape:
        raise ValueError(
            "Action shape mismatch for record building: "
            f"pred={pred_action_raw.shape} vs gt={gt_action_raw.shape}"
        )

    diff = pred_action_raw - gt_action_raw
    abs_err = np.abs(diff)
    sq_err = diff ** 2
    num_steps, num_dims = abs_err.shape
    records: list[dict] = []
    for time_step in range(num_steps):
        for dim in range(num_dims):
            records.append(
                {
                    "sample_idx": int(sample_idx),
                    "time_step": int(time_step),
                    "dim": int(dim),
                    "pred_action": float(pred_action_raw[time_step, dim]),
                    "gt_action": float(gt_action_raw[time_step, dim]),
                    "signed_error": float(diff[time_step, dim]),
                    "abs_error": float(abs_err[time_step, dim]),
                    "sq_error": float(sq_err[time_step, dim]),
                }
            )
    return records


def _aggregate_action_error_records(records: list[dict]) -> list[dict]:
    """Aggregate error records by time step and action dimension."""
    grouped_errors: dict[tuple[int, int], list[float]] = defaultdict(list)
    for record in records:
        grouped_errors[(int(record["time_step"]), int(record["dim"]))].append(float(record["abs_error"]))

    summary_rows: list[dict] = []
    for (time_step, dim), values in sorted(grouped_errors.items()):
        values_np = np.asarray(values, dtype=np.float64)
        summary_rows.append(
            {
                "time_step": int(time_step),
                "dim": int(dim),
                "count": int(values_np.size),
                "mean_abs_error": float(values_np.mean()),
                "max_abs_error": float(values_np.max()),
                "p50_abs_error": float(np.percentile(values_np, 50)),
                "p75_abs_error": float(np.percentile(values_np, 75)),
                "p90_abs_error": float(np.percentile(values_np, 90)),
                "p95_abs_error": float(np.percentile(values_np, 95)),
                "p99_abs_error": float(np.percentile(values_np, 99)),
            }
        )
    return summary_rows


def _aggregate_sample_error_records(records: list[dict]) -> list[dict]:
    """Aggregate error records by sample, with early/late-window diagnostics."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        grouped[int(record["sample_idx"])].append(record)

    summary_rows: list[dict] = []
    for sample_idx, sample_records in sorted(grouped.items()):
        abs_all = np.asarray([float(r["abs_error"]) for r in sample_records], dtype=np.float64)
        early_vals = np.asarray(
            [float(r["abs_error"]) for r in sample_records if int(r["time_step"]) <= 7],
            dtype=np.float64,
        )
        late_vals = np.asarray(
            [float(r["abs_error"]) for r in sample_records if int(r["time_step"]) >= 24],
            dtype=np.float64,
        )
        summary_rows.append(
            {
                "sample_idx": int(sample_idx),
                "count": int(abs_all.size),
                "mean_abs_error": float(abs_all.mean()),
                "p75_abs_error": float(np.percentile(abs_all, 75)),
                "p90_abs_error": float(np.percentile(abs_all, 90)),
                "p95_abs_error": float(np.percentile(abs_all, 95)),
                "p99_abs_error": float(np.percentile(abs_all, 99)),
                "max_abs_error": float(abs_all.max()),
                "early_mean_abs_error_t_le_7": float(early_vals.mean()) if early_vals.size > 0 else np.nan,
                "late_mean_abs_error_t_ge_24": float(late_vals.mean()) if late_vals.size > 0 else np.nan,
                "late_p90_abs_error_t_ge_24": float(np.percentile(late_vals, 90)) if late_vals.size > 0 else np.nan,
                "late_p95_abs_error_t_ge_24": float(np.percentile(late_vals, 95)) if late_vals.size > 0 else np.nan,
            }
        )
    return summary_rows


def _save_sample_error_rankings(records: list[dict], all_results: list[dict], output_dir: str) -> dict:
    """Save per-sample error ranking tables for outlier analysis."""
    rankings_dir = os.path.join(output_dir, "action_error_sample_rankings")
    os.makedirs(rankings_dir, exist_ok=True)

    sample_rows = _aggregate_sample_error_records(records)
    metrics_by_sample = {
        int(r.get("sample_idx", -1)): r.get("action_error", {}) for r in all_results
    }

    enriched_rows: list[dict] = []
    for row in sample_rows:
        sample_idx = int(row["sample_idx"])
        action_error = metrics_by_sample.get(sample_idx, {})
        row_out = dict(row)
        row_out["summary_mae_global"] = float(action_error.get("mae_global", np.nan))
        row_out["summary_rmse_global"] = float(action_error.get("rmse_global", np.nan))
        row_out["summary_max_abs_error"] = float(action_error.get("max_abs_error", np.nan))
        enriched_rows.append(row_out)

    base_fields = [
        "sample_idx",
        "count",
        "mean_abs_error",
        "p75_abs_error",
        "p90_abs_error",
        "p95_abs_error",
        "p99_abs_error",
        "max_abs_error",
        "early_mean_abs_error_t_le_7",
        "late_mean_abs_error_t_ge_24",
        "late_p90_abs_error_t_ge_24",
        "late_p95_abs_error_t_ge_24",
        "summary_mae_global",
        "summary_rmse_global",
        "summary_max_abs_error",
    ]

    all_samples_path = os.path.join(rankings_dir, "sample_error_rankings_all.csv")
    _write_csv_rows(all_samples_path, enriched_rows, base_fields)

    sorted_specs = [
        ("p95_abs_error", "sample_error_rankings_by_p95_desc.csv"),
        ("p90_abs_error", "sample_error_rankings_by_p90_desc.csv"),
        ("max_abs_error", "sample_error_rankings_by_max_desc.csv"),
        ("late_p95_abs_error_t_ge_24", "sample_error_rankings_by_late_p95_desc.csv"),
    ]
    sorted_paths: dict[str, str] = {}
    for key, filename in sorted_specs:
        sorted_rows = sorted(
            enriched_rows,
            key=lambda r: float(r.get(key, np.nan)),
            reverse=True,
        )
        path = os.path.join(rankings_dir, filename)
        _write_csv_rows(path, sorted_rows, base_fields)
        sorted_paths[key] = path

    return {
        "rankings_dir": rankings_dir,
        "all_samples_path": all_samples_path,
        "sorted_paths": sorted_paths,
    }


def _write_csv_rows(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_action_error_summary_outputs(records: list[dict], output_dir: str) -> dict:
    """Save detailed records, aggregated CSV, and matplotlib SVG plots."""
    records_path = os.path.join(output_dir, "action_error_records.csv")
    summary_path = os.path.join(output_dir, "action_error_summary.csv")
    plot_dir = os.path.join(output_dir, "action_error_plots")
    os.makedirs(plot_dir, exist_ok=True)

    record_fieldnames = [
        "sample_idx",
        "time_step",
        "dim",
        "pred_action",
        "gt_action",
        "signed_error",
        "abs_error",
        "sq_error",
    ]
    _write_csv_rows(records_path, records, record_fieldnames)

    summary_rows = _aggregate_action_error_records(records)
    summary_fieldnames = [
        "time_step",
        "dim",
        "count",
        "mean_abs_error",
        "max_abs_error",
        "p50_abs_error",
        "p75_abs_error",
        "p90_abs_error",
        "p95_abs_error",
        "p99_abs_error",
    ]
    _write_csv_rows(summary_path, summary_rows, summary_fieldnames)

    if len(summary_rows) == 0:
        raise ValueError("No action error records available for summary plotting.")

    time_steps = sorted({int(row["time_step"]) for row in summary_rows})
    dims = sorted({int(row["dim"]) for row in summary_rows})
    stat_keys = [
        ("mean_abs_error", "Mean absolute error"),
        ("max_abs_error", "Max absolute error"),
        ("p50_abs_error", "P50 absolute error"),
        ("p75_abs_error", "P75 absolute error"),
        ("p90_abs_error", "P90 absolute error"),
        ("p95_abs_error", "P95 absolute error"),
        ("p99_abs_error", "P99 absolute error"),
    ]
    stat_label_prefix = {
        "mean_abs_error": "mean",
        "max_abs_error": "max",
        "p50_abs_error": "p50",
        "p75_abs_error": "p75",
        "p90_abs_error": "p90",
        "p95_abs_error": "p95",
        "p99_abs_error": "p99",
    }

    summary_lookup = {
        (int(row["time_step"]), int(row["dim"])): row for row in summary_rows
    }
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(dims))]
    plot_paths: dict[str, str] = {}

    for stat_key, stat_label in stat_keys:
        matrix = np.full((len(time_steps), len(dims)), np.nan, dtype=np.float32)
        for t_idx, time_step in enumerate(time_steps):
            for d_idx, dim in enumerate(dims):
                row = summary_lookup.get((time_step, dim))
                if row is not None:
                    matrix[t_idx, d_idx] = float(row[stat_key])

        # Dynamically size width by number of time steps so long horizons (e.g. 65) fit well.
        # Chooses at least 12 inches, otherwise 0.25 inch per time step plus padding.
        fig_width = max(12.0, 0.25 * len(time_steps) + 4.0)
        fig_height = 6.5
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        x = np.arange(len(time_steps), dtype=np.float32)
        group_width = 0.82
        bar_width = group_width / max(len(dims), 1)
        offsets = (np.arange(len(dims), dtype=np.float32) - (len(dims) - 1) / 2.0) * bar_width

        for dim_idx, dim in enumerate(dims):
            ax.bar(
                x + offsets[dim_idx],
                matrix[:, dim_idx],
                width=bar_width,
                color=colors[dim_idx],
                label=f"dim {dim}",
                linewidth=0,
            )

        ax.set_title(f"Action absolute error by time step ({stat_label})")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Absolute error")
        # Fix y-axis maximum as requested by the user
        ax.set_ylim(0.0, 0.05)
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in time_steps], rotation=0)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(ncols=2, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])

        plot_name = f"action_error_time_step_{stat_label_prefix[stat_key]}.svg"
        plot_path = os.path.join(plot_dir, plot_name)
        fig.savefig(plot_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        plot_paths[stat_key] = plot_path

    return {
        "records_csv_path": records_path,
        "summary_csv_path": summary_path,
        "plot_dir": plot_dir,
        "plot_paths": plot_paths,
    }


def _align_proprio_length(proprio: torch.Tensor, target_len: int) -> torch.Tensor:
    """Align proprio temporal length to target by truncation or last-value padding."""
    if proprio.ndim != 2:
        raise ValueError(f"Expected proprio shape [T, D], got {tuple(proprio.shape)}")
    cur_len = int(proprio.shape[0])
    if cur_len == target_len:
        return proprio
    if cur_len > target_len:
        return proprio[:target_len]
    pad_count = target_len - cur_len
    last_row = proprio[-1:].repeat(pad_count, 1)
    return torch.cat([proprio, last_row], dim=0)


def _save_action_error_visualizations(
    pred_action_raw: np.ndarray,
    gt_action_raw: np.ndarray,
    sample_idx: int,
    output_dir: str,
) -> dict:
    """Save action error heatmap and per-dim MAE bar chart (PIL-based, no extra deps)."""
    diff = pred_action_raw - gt_action_raw  # [T, D]
    abs_err = np.abs(diff)
    t_steps, d_dim = abs_err.shape

    # Heatmap: x-axis is time, y-axis is action dimension
    max_val = float(abs_err.max())
    denom = max(max_val, 1e-8)
    norm = (abs_err / denom).clip(0.0, 1.0)
    r = (norm * 255.0).astype(np.uint8)
    g = ((1.0 - norm) * 255.0).astype(np.uint8)
    b = np.zeros_like(r, dtype=np.uint8)
    heatmap_rgb = np.stack([r, g, b], axis=-1).transpose(1, 0, 2)  # [D, T, 3]
    cell_w, cell_h = 12, 24  # pixels per time step / per dim
    heat_w, heat_h = t_steps * cell_w, d_dim * cell_h
    heatmap_img = Image.fromarray(heatmap_rgb, mode="RGB").resize(
        (heat_w, heat_h),
        resample=Image.NEAREST,
    )
    # Add margins for labels: left=60 (dim names), bottom=20 (time ticks), top=20 (title row)
    margin_left, margin_bottom, margin_top = 60, 20, 22
    canvas_w = margin_left + heat_w
    canvas_h = margin_top + heat_h + margin_bottom
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    canvas.paste(heatmap_img, (margin_left, margin_top))
    draw = ImageDraw.Draw(canvas)
    # Title
    draw.text((margin_left, 4), f"Action Error Heatmap  (red=high, green=low, max={max_val:.4f})", fill=(40, 40, 40))
    # Y-axis: dim labels on the left of each row
    for d in range(d_dim):
        y_center = margin_top + d * cell_h + cell_h // 2 - 4
        draw.text((2, y_center), f"dim{d}", fill=(20, 20, 20))
    # X-axis: time tick every 4 steps
    tick_every = max(1, t_steps // 8)
    for t in range(0, t_steps, tick_every):
        x = margin_left + t * cell_w
        draw.line([(x, margin_top + heat_h), (x, margin_top + heat_h + 4)], fill=(80, 80, 80))
        draw.text((x, margin_top + heat_h + 5), str(t), fill=(40, 40, 40))
    # Last tick
    x_last = margin_left + (t_steps - 1) * cell_w
    draw.line([(x_last, margin_top + heat_h), (x_last, margin_top + heat_h + 4)], fill=(80, 80, 80))
    draw.text((x_last, margin_top + heat_h + 5), str(t_steps - 1), fill=(40, 40, 40))
    heatmap_path = os.path.join(output_dir, "action_error_heatmap.png")
    canvas.save(heatmap_path)

    # Per-dimension MAE bar chart
    mae_per_dim = abs_err.mean(axis=0)
    w, h = 960, 60 + 36 * d_dim
    chart = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(chart)
    draw.text((20, 10), "Per-dimension MAE (dims 0-6 in rad, dim 7 gripper scaled to [0,1])", fill=(0, 0, 0))
    bar_left = 220
    bar_right = w - 40
    bar_max_width = bar_right - bar_left
    max_mae = float(mae_per_dim.max()) if d_dim > 0 else 1.0
    max_mae = max(max_mae, 1e-8)
    for d in range(d_dim):
        y = 40 + d * 36
        width = int((float(mae_per_dim[d]) / max_mae) * bar_max_width)
        draw.text((20, y), f"dim {d}", fill=(20, 20, 20))
        draw.rectangle([bar_left, y + 6, bar_left + width, y + 26], fill=(230, 90, 70))
        draw.rectangle([bar_left, y + 6, bar_right, y + 26], outline=(180, 180, 180), width=1)
        draw.text((bar_right + 8, y), f"{float(mae_per_dim[d]):.6f}", fill=(20, 20, 20))
    bar_path = os.path.join(output_dir, "action_error_mae_bar.png")
    chart.save(bar_path)

    return {
        "action_error_heatmap_path": heatmap_path,
        "action_error_mae_bar_path": bar_path,
    }


def _run_joint_video_infer(
    model,
    *,
    input_image: torch.Tensor,
    prompt: Optional[str],
    context: Optional[torch.Tensor],
    context_mask: Optional[torch.Tensor],
    proprio_input: Optional[torch.Tensor],
    action_for_video: Optional[torch.Tensor],
    action_horizon: int,
    num_video_frames: int,
    num_inference_steps: int,
    seed: int,
    device: str,
) -> list[Image.Image]:
    """Run model.infer_joint and return predicted video frames."""
    infer_kwargs = {
        "prompt": None,
        "input_image": input_image.unsqueeze(0).to(device=device, dtype=model.torch_dtype),
        "num_video_frames": int(num_video_frames),
        "action_horizon": int(action_horizon),
        "action": action_for_video,
        "proprio": proprio_input.to(device=device, dtype=model.torch_dtype) if proprio_input is not None else None,
        "num_inference_steps": int(num_inference_steps),
        "seed": int(seed),
        "tiled": False,
        "test_action_with_infer_action": False,
    }
    if context is not None and context_mask is not None:
        infer_kwargs["context"] = context.unsqueeze(0).to(device=device, dtype=model.torch_dtype)
        infer_kwargs["context_mask"] = context_mask.unsqueeze(0).to(device=device)
    else:
        infer_kwargs["prompt"] = prompt

    joint_out = model.infer_joint(**infer_kwargs)
    return joint_out["video"]


def run_single_sample(
    model,
    sample: dict,
    processor,
    cfg: DictConfig,
    sample_idx: int,
    output_dir: str,
    error_records: list[dict],
    inference_times: list[float],
) -> dict:
    """Run inference on a single dataset sample and save results."""
    infer_cfg = cfg.inference
    device = str(infer_cfg.device)
    num_inference_steps = int(infer_cfg.num_inference_steps)
    seed = int(infer_cfg.seed)
    skip_video_generation = bool(infer_cfg.get("skip_video_generation", False))
    
    # Create per-sample output subfolder
    sample_output_dir = os.path.join(output_dir, f"sample_{sample_idx:06d}")
    os.makedirs(sample_output_dir, exist_ok=True)
    
    # Log configuration warnings for reference
    config_diagnostics = {}

    video = sample["video"]  # [C, T, H, W] in [-1, 1]
    action_gt = sample["action"]  # [T_action, action_dim]
    proprio = sample["proprio"]  # [T_action, state_dim] or None
    context = sample.get("context", None)
    context_mask = sample.get("context_mask", None)
    prompt = sample.get("prompt", None)

    # First frame as input image: [3, H, W]
    input_image = video[:, 0]  # [C, H, W]

    # Action horizon from GT and optional override
    gt_action_horizon = int(action_gt.shape[0])
    action_horizon = gt_action_horizon
    action_horizon_override = infer_cfg.get("action_horizon_override", None)
    if action_horizon_override is not None:
        action_horizon = int(action_horizon_override)

    # Proprio: use first timestep [D]
    proprio_input = None
    if proprio is not None:
        proprio_input = proprio[0]  # [D]

    # Build inference kwargs
    infer_kwargs = {
        "prompt": None,
        "input_image": input_image.unsqueeze(0).to(device=device, dtype=model.torch_dtype),
        "action_horizon": action_horizon,
        "proprio": proprio_input.to(device=device, dtype=model.torch_dtype) if proprio_input is not None else None,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "tiled": False,
    }

    if context is not None and context_mask is not None:
        infer_kwargs["context"] = context.unsqueeze(0).to(device=device, dtype=model.torch_dtype)
        infer_kwargs["context_mask"] = context_mask.unsqueeze(0).to(device=device)
    else:
        infer_kwargs["prompt"] = prompt

    # Run action-only inference and measure time
    t0 = time.perf_counter()
    pred = model.infer_action(**infer_kwargs)
    t1 = time.perf_counter()
    infer_time = float(t1 - t0)
    inference_times.append(infer_time)
    logger.info("Sample %d action inference time: %.4f s", sample_idx, infer_time)
    pred_action = pred["action"]  # [T_action, action_dim]

    # Denormalize predicted action
    if proprio is None:
        raise ValueError("sample['proprio'] is required for action denormalization and video conditioning.")
    pred_proprio = _align_proprio_length(proprio, int(pred_action.shape[0]))
    gt_proprio = _align_proprio_length(proprio, int(action_gt.shape[0]))
    pred_action_raw = _denormalize_action(pred_action, pred_proprio, processor)  # [1, T_action, action_dim]
    gt_action_raw = _denormalize_action(action_gt, gt_proprio, processor)  # [1, T_action, action_dim]

    # x1 gripper is stored on a 0-100 scale; normalize it to [0, 1] so it does not dominate the action error plots.
    pred_action_raw[..., -1] = pred_action_raw[..., -1] / 100.0
    gt_action_raw[..., -1] = gt_action_raw[..., -1] / 100.0

    # Action error metrics + visualizations
    pred_raw_2d = pred_action_raw.squeeze(0)
    gt_raw_2d = gt_action_raw.squeeze(0)
    overlap_steps = min(int(pred_raw_2d.shape[0]), int(gt_raw_2d.shape[0]))
    if overlap_steps <= 0:
        raise ValueError("No temporal overlap between predicted and GT actions for metric computation.")
    pred_eval = pred_raw_2d[:overlap_steps]
    gt_eval = gt_raw_2d[:overlap_steps]
    action_error_metrics = _compute_action_error_metrics(pred_eval, gt_eval)
    action_error_metrics["pred_steps"] = int(pred_raw_2d.shape[0])
    action_error_metrics["gt_steps"] = int(gt_raw_2d.shape[0])
    action_error_metrics["overlap_steps"] = int(overlap_steps)
    error_records.extend(_build_action_error_records(pred_eval, gt_eval, sample_idx=sample_idx))
    action_viz_paths = _save_action_error_visualizations(
        pred_eval,
        gt_eval,
        sample_idx=sample_idx,
        output_dir=sample_output_dir,
    )

    result = {
        "sample_idx": sample_idx,
        "action_horizon": action_horizon,
        "prompt": prompt,
        "action_inference_time_s": infer_time,
        "action_error": action_error_metrics,
        "config_diagnostics": config_diagnostics,
        **action_viz_paths,
    }

    # Save actions
    if infer_cfg.save_actions:
        actions_path = os.path.join(sample_output_dir, "actions.npz")
        np.savez(
            actions_path,
            pred_action=pred_raw_2d,
            gt_action=gt_raw_2d,
            pred_action_normalized=pred_action.numpy() if isinstance(pred_action, torch.Tensor) else pred_action,
            gt_action_normalized=action_gt.numpy() if isinstance(action_gt, torch.Tensor) else action_gt,
        )
        result["actions_path"] = actions_path
        logger.info("  Saved actions -> %s", actions_path)

    action_metrics_path = os.path.join(sample_output_dir, "action_metrics.json")
    with open(action_metrics_path, "w", encoding="utf-8") as f:
        json.dump(action_error_metrics, f, ensure_ascii=True, indent=2)
    result["action_metrics_path"] = action_metrics_path
    logger.info("  Saved action metrics -> %s", action_metrics_path)
    
    if config_diagnostics:
        diagnostics_path = os.path.join(sample_output_dir, "config_diagnostics.json")
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(config_diagnostics, f, ensure_ascii=True, indent=2)
        result["config_diagnostics_path"] = diagnostics_path

    # Save GT video for reference
    if infer_cfg.save_gt_video and not skip_video_generation:
        gt_frames = _video_tensor_to_pil_frames(video)
        gt_video_path = os.path.join(sample_output_dir, "gt_video.mp4")
        save_mp4(gt_frames, gt_video_path, fps=8)
        result["gt_video_path"] = gt_video_path
        logger.info("  Saved GT video -> %s", gt_video_path)

    # Optional video prediction: based on GT action / predicted action / no action
    video_cfg = infer_cfg.get("video_prediction", {})
    if (not skip_video_generation) and bool(video_cfg.get("enabled", True)):
        fps = int(video_cfg.get("fps", 8))
        # Prioritize explicit num_video_frames config over sample GT length
        num_video_frames_cfg = video_cfg.get("num_video_frames", None)
        if num_video_frames_cfg is not None:
            num_video_frames = int(num_video_frames_cfg)
        else:
            num_video_frames = int(video.shape[1])

        # Check if num_video_frames exceeds GT video length
        if num_video_frames > (int(video.shape[1] - 1) * 4 + 1):
            logger.warning(
                "  [WARN] num_video_frames (%d) > GT length (%d). "
                "The generated video will extrapolate beyond the training distribution.",
                num_video_frames,
                (int(video.shape[1] - 1) * 4 + 1),
            )
            config_diagnostics["video_extrapolation"] = {
                "num_video_frames": num_video_frames,
                "gt_video_frames": int(video.shape[1]),
                "extrapolation_frames": num_video_frames - (int(video.shape[1] - 1) * 4 + 1),
            }

        gt_frames = _video_tensor_to_pil_frames(video)
        action_for_gt = action_gt.to(device=device, dtype=model.torch_dtype)
        action_for_pred = pred_action.to(device=device, dtype=model.torch_dtype)

        def _save_joint_mode(mode_name: str, action_for_video: Optional[torch.Tensor]):
            try:
                pred_frames = _run_joint_video_infer(
                    model,
                    input_image=input_image,
                    prompt=prompt,
                    context=context,
                    context_mask=context_mask,
                    proprio_input=proprio_input,
                    action_for_video=action_for_video,
                    action_horizon=action_horizon,
                    num_video_frames=num_video_frames,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    device=device,
                )
                # Stitch GT on top, prediction on bottom
                stitched = _stitch_frames_vert(gt_frames, pred_frames)
                if len(stitched) == 0:
                    raise ValueError("Stitched video has zero frames.")
                out_path = os.path.join(sample_output_dir, f"{mode_name}_gt_top_pred_bottom.mp4")
                save_mp4(stitched, out_path, fps=fps)
                result[f"video_{mode_name}_path"] = out_path
                logger.info("  Saved stitched video (%s, GT top / pred bottom) -> %s", mode_name, out_path)
            except Exception as e:
                logger.warning("  Skipped video mode %s due to error: %s", mode_name, e)
                result[f"video_{mode_name}_error"] = str(e)

        if bool(video_cfg.get("with_gt_action", True)):
            _save_joint_mode("with_gt_action", action_for_gt)
        if bool(video_cfg.get("with_pred_action", True)):
            _save_joint_mode("with_pred_action", action_for_pred)
        if bool(video_cfg.get("without_action", True)):
            _save_joint_mode("without_action", None)

    return result


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="infer_x1_insert")
def main(cfg: DictConfig):
    setup_logging(log_level=logging.INFO)
    infer_cfg = cfg.inference
    device = str(infer_cfg.device)
    output_dir = str(infer_cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Save resolved config
    config_save_path = os.path.join(output_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), f)
    logger.info("Saved config -> %s", config_save_path)

    # Load model
    logger.info("Loading model...")
    model = _load_model(cfg, device=device)
    logger.info("Model loaded on %s", device)

    # Compile model
    logger.info("Compiling model...")
    model = torch.compile(model, mode="reduce-overhead")
    logger.info("Model compiled.")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = _load_dataset(cfg)
    processor = dataset.lerobot_dataset.processor
    logger.info("Dataset loaded: %d samples", len(dataset))

    error_records: list[dict] = []
    inference_times: list[float] = []
    
    # Select sample indices
    num_samples = int(infer_cfg.num_samples)
    if infer_cfg.sample_indices is not None:
        sample_indices = list(infer_cfg.sample_indices)
    else:
        rng = np.random.RandomState(int(infer_cfg.seed))
        sample_indices = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False).tolist()

    logger.info("Running inference on %d samples: %s", len(sample_indices), sample_indices)

    all_results = []
    for i, idx in enumerate(sample_indices):
        logger.info("[%d/%d] Sample index=%d", i + 1, len(sample_indices), idx)
        sample = dataset[idx]
        result = run_single_sample(
            model=model,
            sample=sample,
            processor=processor,
            cfg=cfg,
            sample_idx=idx,
            output_dir=output_dir,
            error_records=error_records,
            inference_times=inference_times,
        )
        all_results.append(result)

    error_summary_paths = _save_action_error_summary_outputs(error_records, output_dir)
    logger.info("Saved detailed action error records -> %s", error_summary_paths["records_csv_path"])
    logger.info("Saved aggregated action error summary -> %s", error_summary_paths["summary_csv_path"])
    logger.info("Saved action error plots -> %s", error_summary_paths["plot_dir"])

    sample_ranking_paths = _save_sample_error_rankings(error_records, all_results, output_dir)
    logger.info("Saved per-sample error rankings -> %s", sample_ranking_paths["rankings_dir"])

    for result in all_results:
        result["action_error_records_csv_path"] = error_summary_paths["records_csv_path"]
        result["action_error_summary_csv_path"] = error_summary_paths["summary_csv_path"]
        result["action_error_plot_dir"] = error_summary_paths["plot_dir"]
        result["action_error_plot_paths"] = error_summary_paths["plot_paths"]
        result["action_error_sample_rankings_dir"] = sample_ranking_paths["rankings_dir"]
        result["action_error_sample_rankings_all_csv_path"] = sample_ranking_paths["all_samples_path"]
        result["action_error_sample_rankings_sorted_csv_paths"] = sample_ranking_paths["sorted_paths"]

    if len(inference_times) > 0:
        avg_time = float(sum(inference_times) / len(inference_times))
        logger.info(
            "Average action inference time over %d samples: %.4f s",
            len(inference_times),
            avg_time,
        )
    else:
        logger.info("No action inference timings were recorded.")

    # Save summary
    summary_path = os.path.join(output_dir, "inference_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    logger.info("Inference complete. Summary -> %s", summary_path)


if __name__ == "__main__":
    main()
