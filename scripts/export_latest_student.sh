#!/bin/bash
# This script automates the process of loading the trained distilled Student model,
# exporting it to TorchScript (policy.pt) and ONNX (policy.onnx), and copying them 
# to the /home/shw/go1_real/model directory for real-world deployment.

set -e

# 1. Basic Environment configuration
export GO1_PHASE="student"
LOG_DIR="/home/shw/go1_peg/scripts/rsl_rl/logs/rsl_rl/unitree_go1_rough_student/2026-05-12_23-16-26_student_v9_seed42"
CHECKPOINT="$LOG_DIR/model_4999.pt"
EXPORT_DIR="$LOG_DIR/exported"

echo "========================================================"
echo "🚀 Starting automated policy export wrapper..."
echo "📂 Source Checkpoint: $CHECKPOINT"
echo "📂 Output Target: /home/shw/go1_real/model/"
echo "========================================================"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint file not found at $CHECKPOINT"
    exit 1
fi

# Clear older exports
rm -rf "$EXPORT_DIR"

# 2. Start play.py in the background with headless mode
# We use '--headless' and '--num_envs 1' to speed up and avoid GUI dependency.
echo "🔄 Launching Isaac Sim headless wrapper in background..."
export CONDA_DEFAULT_ENV="isaac"
export CONDA_PREFIX="/home/shw/miniconda3/envs/isaac"
export TERM=xterm

/home/shw/miniconda3/envs/isaac/bin/python /home/shw/go1_peg/scripts/rsl_rl/play.py \
    --task Template-Go1-Lab-v0 \
    --agent rsl_rl_distill_cfg_entry_point \
    --checkpoint "$CHECKPOINT" \
    --headless \
    --num_envs 1 &
PLAY_PID=$!

# Setup cleanup trap
cleanup() {
    echo "🧹 Cleaning up background processes (PID $PLAY_PID)..."
    kill -9 $PLAY_PID 2>/dev/null || true
}
trap cleanup EXIT SIGINT SIGTERM

echo "📡 Background Process PID: $PLAY_PID. Waiting for export..."

# 3. Polling loop to detect file export
EXPORTED=false
for i in {1..300}; do
    # Check if the process is still running
    if ! kill -0 $PLAY_PID 2>/dev/null; then
        echo "⚠️ Background process exited unexpectedly early."
        break
    fi

    if [ -f "$EXPORT_DIR/policy.pt" ] && [ -f "$EXPORT_DIR/policy.onnx" ]; then
        echo "✅ Success: Exported policy files generated!"
        EXPORTED=true
        break
    fi
    
    if (( i % 5 == 0 )); then
        echo "⏳ Waiting for export... (${i}s elapsed)"
    fi
    sleep 1
done

# 4. Shutdown background simulator
cleanup
# Unset trap since we already ran cleanup
trap - EXIT SIGINT SIGTERM

# 5. Check outcomes and copy
if [ "$EXPORTED" = true ]; then
    echo "📦 Copying models into go1_real workspace..."
    mkdir -p /home/shw/go1_real/model
    cp "$EXPORT_DIR/policy.pt" /home/shw/go1_real/model/policy.pt
    cp "$EXPORT_DIR/policy.onnx" /home/shw/go1_real/model/policy.onnx
    
    echo "✨ All models copied successfully!"
    echo "📁 Output listing:"
    ls -lh /home/shw/go1_real/model/
    echo "========================================================"
    echo "🎉 EXPORT COMPLETE! Ready for deploy_policy.py"
    echo "========================================================"
else
    echo "❌ Export failed or timed out after 5 minutes."
    exit 1
fi
