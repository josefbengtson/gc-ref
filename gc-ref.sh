#!/bin/bash
exp_name="No_Experiment_Name"
repeat=1
batch_size=1
num_poses=1
x_end=0.2
rotation_angle=10
lr=0.025
num_opt_steps=35
alpha_rgb=2.5
input_image_idx=0
scene_name="room"
seed=0
num_matches=50000
ddim_steps=50
chkpt_name="megascenes"
certainty_threshold=0.15
az_start=5
az_end=25
az_step=5

while [[ $# -gt 0 ]]; do
    case "$1" in
            --exp_name)
            exp_name=$2
            shift 2
            ;;
            --repeat)
            repeat=$2
            shift 2
            ;;
            --batch_size)
            batch_size=$2
            shift 2
            ;;
            --n)
            num_poses=$2
            shift 2
            ;;
            --x_end)
            x_end=$2
            shift 2
            ;;
            --rotation_angle)
            rotation_angle=$2
            shift 2
            ;;
            --lr)
            lr=$2
            shift 2
            ;;
            --num_opt_steps)
            num_opt_steps=$2
            shift 2
            ;;
            --alpha_rgb)
            alpha_rgb=$2
            shift 2
            ;;
            --input_image_idx)
            input_image_idx=$2
            shift 2
            ;;
            --scene_name)
              scene_name=$2
              shift 2
              ;;
            --seed)
              seed=$2
              shift 2
              ;;
            --num_matches)
              num_matches=$2
              shift 2
              ;;
            --ddim_steps)
              ddim_steps=$2
              shift 2
              ;;
            --chkpt_name)
              chkpt_name=$2
              shift 2
              ;;
            --certainty_threshold)
              certainty_threshold=$2
              shift 2
              ;;
            --split_type)
              split_type=$2
              shift 2
              ;;
            --az_start)
              az_start=$2
              shift 2
              ;;
            --az_end)
              az_end=$2
              shift 2
              ;;
            --az_step)
              az_step=$2
              shift 2
              ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done


# If chkpt_name="megascenes", then the config file is in the following path
if [ "$chkpt_name" == "megascenes" ]; then
  echo "Using MegaScenes checkpoint"
  export CONFIG_PATH=/mimer/NOBACKUP/groups/snic2022-6-266/josef/MegaScenes/configs/warp_plus_pose/
  export chktpt_number=112000
elif [ "$chkpt_name" == "zeronvs_finetune" ]; then
  echo "Using ZeroNVS finetuned checkpoint"
  export CONFIG_PATH=/mimer/NOBACKUP/groups/snic2022-6-266/josef/MegaScenes/configs/zeronvs_finetune/
  export chktpt_number=90000
elif [ "$chkpt_name" == "zeronvs_original" ]; then
  echo "Using ZeroNVS original checkpoint"
  export CONFIG_PATH=/mimer/NOBACKUP/groups/snic2022-6-266/josef/MegaScenes/configs/zeronvs_original/
  export chktpt_number=0
else
  echo "Unknown checkpoint name: $chkpt_name"
  exit 1
fi


export full_exp_name=$exp_name


echo "Running experiment: $exp_name"
echo "Full experiment name: $full_exp_name"
echo "Number Repeat: $repeat"
echo "Batch size: $batch_size"
echo "Number of poses: $num_poses"
echo "X end: $x_end"
echo "Rotation angle: $rotation_angle"
echo "Learning rate: $lr"
echo "Number of optimization steps: $num_opt_steps"
echo "Alpha rgb: $alpha_rgb"
echo "Input image index: $input_image_idx"
echo "Scene name: $scene_name"
echo "Seed: $seed"
echo "Number of matches: $num_matches"
echo "Number of ddim steps: $ddim_steps"
echo "Checkpoint name: $chkpt_name"
echo "Certainty threshold: $certainty_threshold"
echo "Split type: $split_type"
echo "Azimuth start: $az_start"
echo "Azimuth end: $az_end"
echo "Azimuth step: $az_step"


# For megascenes
export SAVE_PATH=/mimer/NOBACKUP/groups/naiss2025-23-200/MegaScenes/Results/megascenes/$exp_name/scenes/$scene_name
export IMAGE_PATH=/mimer/NOBACKUP/groups/snic2022-6-266/josef/MegaScenes/data/megascenes/$split_type/scenes/$scene_name/image1.jpg
export INTRINSICS_PATH=/mimer/NOBACKUP/groups/snic2022-6-266/josef/MegaScenes/data/megascenes/$split_type/scenes/$scene_name/intrinsics1.npz

echo "Saving to: $SAVE_PATH"
echo "Image path: $IMAGE_PATH"


if [ "$chkpt_name" == "megascenes" ]; then
  python gc-ref.py -e $CONFIG_PATH -r $chktpt_number -i $IMAGE_PATH -s $SAVE_PATH -b "$batch_size" --intrinsics_path $INTRINSICS_PATH --repeat "$repeat" --num_poses "$num_poses" --x_end "$x_end" --rotation_angle "$rotation_angle" --exp_name "$full_exp_name" --lr "$lr" --num_opt_steps "$num_opt_steps" --alpha_rgb "$alpha_rgb" --seed "$seed" --input_image_idx "$input_image_idx" --num_matches "$num_matches" --ddim_steps "$ddim_steps" --scene_name "$scene_name" --certainty_threshold "$certainty_threshold" --split_type "$split_type" --az_start "$az_start" --az_end "$az_end" --az_step "$az_step"
elif [ "$chkpt_name" == "zeronvs_finetune" ]; then
  python gc-ref.py -e $CONFIG_PATH -r $chktpt_number -i $IMAGE_PATH -s $SAVE_PATH -b "$batch_size" --repeat "$repeat" --num_poses "$num_poses" --x_end "$x_end" --rotation_angle "$rotation_angle" --exp_name "$full_exp_name" --lr "$lr" --num_opt_steps "$num_opt_steps" --alpha_rgb "$alpha_rgb" --seed "$seed" --input_image_idx "$input_image_idx" --num_matches "$num_matches" --ddim_steps "$ddim_steps" --scene_name "$scene_name" --certainty_threshold "$certainty_threshold" --split_type "$split_type" --az_start "$az_start" --az_end "$az_end" --az_step "$az_step" -z
elif [ "$chkpt_name" == "zeronvs_original" ]; then
  python gc-ref.py -e $CONFIG_PATH -r $chktpt_number -i $IMAGE_PATH -s $SAVE_PATH -b "$batch_size" --repeat "$repeat" --num_poses "$num_poses" --x_end "$x_end" --rotation_angle "$rotation_angle" --exp_name "$full_exp_name" --lr "$lr" --num_opt_steps "$num_opt_steps" --alpha_rgb "$alpha_rgb" --seed "$seed" --input_image_idx "$input_image_idx" --num_matches "$num_matches" --ddim_steps "$ddim_steps" --scene_name "$scene_name" --certainty_threshold "$certainty_threshold" --split_type "$split_type" --az_start "$az_start" --az_end "$az_end" --az_step "$az_step" -z --ckpt_file
else
  echo "Unknown checkpoint name: $chkpt_name"
  exit 1
fi


# bash ./gc-ref.sh --split_type "test" --num_opt_steps 35 --chkpt_name "megascenes" --exp_name "DevRelease/megascenes/" --alpha 2.5 --certainty_threshold 0.15 --lr 0.025 --seed 1 --az_start 5 --az_end 15 --scene_name 0
