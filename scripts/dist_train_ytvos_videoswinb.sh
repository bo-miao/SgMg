#!/usr/bin/env bash
set -x
cd ..

GPUS='0,1'
PORT=25501
GPUS_PER_NODE=2
CPUS_PER_TASK=6
export CUDA_VISIBLE_DEVICES=${GPUS}
echo "using gpus ${GPUS}, master port ${PORT}."
now=$(date +"%T")
echo "Current time : $now"
echo "Current path : $PWD"

BACKBONE="video_swin_b_p4w7"
BACKBONE_PRETRAINED="./checkpoints/backbones/swin_base_patch244_window877_kinetics600_22k.pth"
OUTPUT_DIR1="./checkpoints/results/SgMg_${BACKBONE}_pretrain"
EXP_NAME1="SgMg_${BACKBONE}_pretrain"
CUDA_VISIBLE_DEVICES=${GPUS} OMP_NUM_THREADS=${CPUS_PER_TASK} torchrun --master_port ${PORT}  --nproc_per_node=${GPUS_PER_NODE} main_pretrain.py \
  --dataset_file all \
  --with_box_refine --binary \
  --output_dir=${OUTPUT_DIR1} \
  --exp_name=${EXP_NAME1} \
  --backbone=${BACKBONE} \
  --backbone_pretrained=${BACKBONE_PRETRAINED} \
  --batch_size 2 \
  --num_frames 1 \
  --epochs 11 --lr_drop 8 10 \


OUTPUT_DIR2="./checkpoints/results/SgMg_${BACKBONE}_finetune"
EXP_NAME2="SgMg_${BACKBONE}_finetune"
CUDA_VISIBLE_DEVICES=${GPUS} OMP_NUM_THREADS=${CPUS_PER_TASK} torchrun --master_port ${PORT}  --nproc_per_node=${GPUS_PER_NODE} main.py \
  --with_box_refine --binary --freeze_text_encoder \
  --output_dir=${OUTPUT_DIR2} \
  --exp_name=${EXP_NAME2} \
  --backbone=${BACKBONE} \
  --backbone_pretrained=${BACKBONE_PRETRAINED} \
  --epochs 6 --lr_drop 3 5 \
  --dataset_file ytvos \
  --pretrained_weights ${OUTPUT_DIR1}"/checkpoint0010.pth" \

