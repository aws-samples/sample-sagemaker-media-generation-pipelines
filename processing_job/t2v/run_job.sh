#!/bin/bash
#
# Usage: run_job.sh --model <ltx|wan> --mode <i2v|t2v>
#
# Arguments:
#   --model   Model to run: ltx (LTX 2.3) or wan (Wan 2.2). Default: ltx
#   --mode    Generation mode: i2v (image-to-video) or t2v (text-to-video). Default: i2v

MODEL="ltx"
MODE="i2v"

while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL="$2"; shift 2 ;;
    --mode)  MODE="$2";  shift 2 ;;
    *)       echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Validate inputs
if [[ "$MODEL" != "ltx" && "$MODEL" != "wan" ]]; then
  echo "Error: --model must be 'ltx' or 'wan', got '$MODEL'"
  exit 1
fi
if [[ "$MODE" != "i2v" && "$MODE" != "t2v" ]]; then
  echo "Error: --mode must be 'i2v' or 't2v', got '$MODE'"
  exit 1
fi

echo "Model: $MODEL, Mode: $MODE"

echo "=== Input directory contents ==="
ls -la /opt/ml/processing/input/ 2>/dev/null || echo "(no /opt/ml/processing/input/)"
ls -la /opt/ml/processing/input/input/ 2>/dev/null || echo "(no /opt/ml/processing/input/input/)"

comfy launch --background
echo "Launching ComfyUI in background..."
sleep 120

cd $HOME
echo "Running $MODEL in $MODE mode"

if [[ "$MODEL" == "ltx" ]]; then
  python3 ltx.py --mode "$MODE"
else
  python3 wan22.py --mode "$MODE"
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
python3 -m common.log_outputs --extensions .mp4 .webm .mkv --metadata --extra model="$MODEL" mode="$MODE"

echo "Script completed successfully."
