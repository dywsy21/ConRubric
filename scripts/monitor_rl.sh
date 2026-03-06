#!/bin/bash
# RL Training Monitor — extracts key metrics from rl_train.log
# Usage: ssh autodl2 'bash /root/autodl-tmp/GRM/scripts/monitor_rl.sh'

LOG="/root/autodl-tmp/GRM/rl_train.log"

echo "========== RL Training Monitor $(date '+%H:%M:%S') =========="
echo ""

# 1. Progress
echo "--- Progress ---"
grep "Training Progress" "$LOG" | tail -1 | sed 's/.*Training Progress: //'
echo ""

# 2. Key metrics table (last N steps)
N=${1:-10}
echo "--- Last $N Steps: Key Metrics ---"
printf "%-5s | %-8s | %-8s | %-8s | %-7s | %-7s | %-8s | %-8s | %-6s | %-6s | %-7s\n" \
  "Step" "Reward" "Rwd_Max" "Rwd_Min" "PG_Loss" "KL_Loss" "Entropy" "GradNrm" "RspLen" "ClipR%" "StepT"
echo "------|----------|----------|----------|---------|---------|----------|----------|--------|--------|--------"

grep "^(TaskRunner.*step:" "$LOG" | tail -$N | while IFS= read -r line; do
  step=$(echo "$line" | grep -oP 'step:\K[0-9]+')
  reward=$(echo "$line" | grep -oP 'critic/score/mean:\K[0-9.]+' | head -1)
  rwd_max=$(echo "$line" | grep -oP 'critic/score/max:\K[0-9.]+' | head -1)
  rwd_min=$(echo "$line" | grep -oP 'critic/score/min:\K[0-9.]+' | head -1)
  pg_loss=$(echo "$line" | grep -oP 'actor/pg_loss:\K-?[0-9.e+-]+' | head -1)
  kl_loss=$(echo "$line" | grep -oP 'actor/kl_loss:\K[0-9.]+' | head -1)
  entropy=$(echo "$line" | grep -oP 'actor/entropy:\K[0-9.]+' | head -1)
  grad_norm=$(echo "$line" | grep -oP 'actor/grad_norm:\K[0-9.]+' | head -1)
  rsp_len=$(echo "$line" | grep -oP 'response_length/mean:\K[0-9.]+' | head -1)
  clip_r=$(echo "$line" | grep -oP 'response_length/clip_ratio:\K[0-9.]+' | head -1)
  step_t=$(echo "$line" | grep -oP 'timing_s/step:\K[0-9.]+' | head -1)

  printf "%-5s | %-8.3f | %-8.3f | %-8.3f | %-7.4f | %-7.4f | %-8.4f | %-8.4f | %-6.0f | %-5.1f%% | %-6.0fs\n" \
    "$step" "$reward" "$rwd_max" "$rwd_min" "$pg_loss" "$kl_loss" "$entropy" "$grad_norm" "$rsp_len" "$(echo "$clip_r * 100" | bc)" "$step_t"
done

echo ""

# 3. Rubric quality samples (latest step)
echo "--- Latest Rubric Samples ---"
grep "\[RubricSample\]" "$LOG" | tail -10

echo ""

# 4. GPU stats
echo "--- GPU Status ---"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | \
  awk -F', ' '{printf "GPU %s: %s/%s MB (%.0f%% util, %s°C)\n", $1, $2, $3, $4, $5}'

echo ""

# 5. Disk usage
echo "--- Disk Usage ---"
df -h /root/autodl-tmp | tail -1 | awk '{print "Data disk:", $3, "used /", $2, "total (" $5 " used)"}'
df -h / | tail -1 | awk '{print "System disk:", $3, "used /", $2, "total (" $5 " used)"}'

# 6. Check for errors
ERR_COUNT=$(grep -c "Traceback\|Error\|OOM\|CUDA error" "$LOG" 2>/dev/null)
if [ "$ERR_COUNT" -gt 0 ]; then
  echo ""
  echo "!!! WARNING: $ERR_COUNT error lines detected !!!"
  grep "Traceback\|Error\|OOM\|CUDA error" "$LOG" | tail -5
fi

echo ""
echo "========== End Monitor =========="
