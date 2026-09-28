set -e
PY=hrc_communication/.venv/Scripts/python.exe
D=hrc_communication/other
# Wait for round 1 (the annotation team's requested configuration) to finish.
# pgrep cannot see Windows processes from Git Bash, so poll the result files.
while [ "$(ls bench/results/loso_N1_full_bg.json bench/results/loso_N2_full_idle.json \
            bench/results/loso_N3_red_bg.json bench/results/loso_N4_red_idle.json \
            2>/dev/null | wc -l)" -lt 4 ]; do sleep 60; done

BASE="--targets peak --aug mirror --data $D/original --aug-dir $D/augmented_mirror --save-models --panels all"
# Round 2: the measured, still-unapplied wins.
#   window 120 -> 240 frames was the largest unused gain (+0.036 on one split)
#   bg loss weight 0.2 -> 0.5 was +0.012
# Epochs are NOT raised: validation macro-F1 peaks at epoch 6 and declines by 12,
# so more epochs would only deepen the train/val gap (0.757 vs 0.451).
$PY bench/loso.py $BASE --win 240 --epochs 10 --tag W1_win240      > bench/logs/W1.log 2>&1
$PY bench/loso.py $BASE --win 240 --epochs 10 --bg-weight 0.5 --tag W2_win240_bg05 > bench/logs/W2.log 2>&1
echo ROUND2_DONE
