#!/bin/bash

num=${1:-10}
comfy launch --background
echo "Launching ComfyUI in background..."
sleep 120

echo "Downloading Models"
cd $COMFY_HOME
comfy model download \
   --url "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" \
   --relative-path "./models/diffusion_models/" \
   --filename "z_image_turbo_bf16.safetensors"

comfy model download \
   --url "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" \
   --relative-path "./models/text_encoders/" \
   --filename "qwen_3_4b.safetensors"

comfy model download \
   --url "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors" \
   --relative-path "./models/vae/" \
   --filename "ae.safetensors"

cd $HOME
echo "Generating $num images"
for i in $(seq 1 $num); do
    echo "Sending workflow $i of $num"
    python3 run_workflow.py --seed $i --workflow image_z_image_turbo.json
done

echo "Starting queue monitoring..."
while true; do
  output=$(python3 is_queue_empty.py)
  if echo "$output" | grep -q "Queue size: 0"; then
    echo "Queue is empty. Stopping comfy..."
    comfy stop
    echo "Comfy stop command executed."
    break
  else
    echo "$output"
    echo "Queue is not empty. Waiting before checking again..."
    sleep 15
  fi
done
echo "Script completed successfully."

