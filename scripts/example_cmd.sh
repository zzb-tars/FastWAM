PATH=/mnt/data/miniconda3/envs/fastwam_train_v0/bin:$PATH GPU_IDS=3,5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash scripts/train_zero2_offload.sh 2 task=libero_uncond_2cam224_1e-4 batch_size=1 gradient_accumulation_steps=8 num_workers=4


GPU_IDS=3,5 bash scripts/train_zero2.sh 2 task=libero_uncond_2cam224_1e-4 batch_size=1 gradient_accumulation_steps=8 num_workers=4


torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4


bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4


# naive
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=8

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8

# naive + task
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.task_suite_name=libero_10 \
  EVALUATION.num_trials=1 \
  MULTIRUN.num_gpus=8

python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8

# single eval
CUDA_VISIBLE_DEVICES=0 \
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  gpu_id=0 \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=1 \
  EVALUATION.output_dir=./evaluate_results/debug_single
