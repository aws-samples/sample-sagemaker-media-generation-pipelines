#!/bin/bash
#
# i2v: Image-to-video generation using LTX and Wan 2.2
#
# Usage: run_job.sh [--model <ltx|wan|both>]
#
# Arguments:
#   --model   Model to run: ltx, wan, or both. Default: both

MODEL="both"

while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL="$2"; shift 2 ;;
    --run)   shift ;;  # legacy arg, ignored
    *)       echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "$MODEL" != "ltx" && "$MODEL" != "wan" && "$MODEL" != "both" ]]; then
  echo "Error: --model must be 'ltx', 'wan', or 'both', got '$MODEL'"
  exit 1
fi

echo "Model: $MODEL, Mode: i2v"

echo "=== Input directory contents ==="
ls -la /opt/ml/processing/input/ 2>/dev/null || echo "(no /opt/ml/processing/input/)"
ls -la /opt/ml/processing/input/input/ 2>/dev/null || echo "(no input/input/)"

comfy launch --background
echo "Launching ComfyUI in background..."
sleep 120

cd $HOME

if [[ "$MODEL" == "ltx" || "$MODEL" == "both" ]]; then
  echo "Running LTX 2.3 in i2v mode..."
  python3 ltx.py --mode i2v
fi

if [[ "$MODEL" == "wan" || "$MODEL" == "both" ]]; then
  echo "Running Wan 2.2 in i2v mode..."
  python3 wan22.py --mode i2v
fi

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

echo "Copying generated videos to SageMaker output directory..."
cp -r "$COMFY_HOME/output/video/." "$LOCAL_OUTPUT_DIR/" 2>/dev/null || true
echo "Files in output dir: $(ls -la $LOCAL_OUTPUT_DIR/)"

echo "Logging generated videos to DynamoDB..."
python3 -m common.log_outputs --extensions .mp4 .webm .mkv --metadata --extra model="$MODEL" mode="i2v"

echo "Script completed successfully."
