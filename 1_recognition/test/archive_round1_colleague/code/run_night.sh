# Overnight runs -- started manually with:   bash bench/run_night.sh
# W1/W2: window 4 s -> 8 s, and bg weight 0.2 -> 0.5. Compared against N1.
# L1:    LSTM with N1's exact settings, for the GRU-vs-LSTM question.
# C1: Screw + Clamp Coupling emphasised in the loss. F1: data-driven 7-panel feature set. S1-S3: trained at 10 fps to match the live loop. Roughly 4 hours in total. Waits for round 1 if it is still running.
set -e
cd "$(dirname "$0")/.."
PY=hrc_communication/.venv/Scripts/python.exe
D=hrc_communication/other
BASE="--targets peak --aug mirror --data $D/original --aug-dir $D/augmented_mirror --save-models"
while [ ! -f bench/results/loso_N6_redvel_idle.json ]; do
  echo "waiting for round 1 to finish..."; sleep 120
done
# F1: the 7 data-driven panels (bench/feat_importance.py), 121 dims, else as N1.
$PY bench/loso.py $BASE --panels selected --tag F1_selected > bench/logs/F1.log 2>&1
# C1: Screw and Clamp Coupling given 2x loss weight, otherwise identical to N1.
$PY bench/loso.py $BASE --panels all --class-weight "Screw=2,Clamp Coupling=2" --tag C1_screw_clamp > bench/logs/C1.log 2>&1
# S-arms: train at the live loop's rate. The corpus is 30 fps but the runtime
# appends one feature vector per ~10 Hz tick, so a model trained on 30 fps
# sequences sees a 3x different time step live. --stride 3 fixes that.
#   S1: 4 s window at 10 fps (40 samples) -- deployment-matched N1
#   S2: 8 s window at 10 fps (80 samples) -- replaces W1 (was 240 frames @30fps)
#   S3: S2 + bg weight 0.5               -- replaces W2
$PY bench/loso.py $BASE --panels all --stride 3 --win 40                  --tag S1_10fps_4s      > bench/logs/S1.log 2>&1
$PY bench/loso.py $BASE --panels all --stride 3 --win 80 --epochs 10      --tag S2_10fps_8s      > bench/logs/S2.log 2>&1
$PY bench/loso.py $BASE --panels all --stride 3 --win 80 --epochs 10 --bg-weight 0.5 --tag S3_10fps_8s_bg05 > bench/logs/S3.log 2>&1
$PY bench/loso.py $BASE --panels all --model lstm                          --tag L1_lstm        > bench/logs/L1.log 2>&1
echo NIGHT_DONE
