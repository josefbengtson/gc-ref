#!/bin/bash

number_images=5
split_type="test"

# Array of numbers to skip
skip_numbers=(100 101 102 103 104 105 106 107 108 109)
start_idx=15
for ((i=start_idx; i<start_idx+number_images; i++)); do
  # Check if the current number is in the skip_numbers array
  if [[ " ${skip_numbers[@]} " =~ " ${i} " ]]; then
    echo "Skipping scene: $i"
    continue
  fi

  scene_name="$i"
  echo "Scene: $scene_name"
  bash ./gc-ref.sh --split_type $split_type --num_opt_steps 35 --chkpt_name "megascenes" --exp_name "DevRelease/gc-ref-small" --scene_name "$scene_name" --alpha_rgb 2.5 --certainty_threshold 0.25 --lr 0.025 --seed 1 --az_start 25 --az_end 25 --az_step 1
done