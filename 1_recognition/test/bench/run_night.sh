# Overnight runs -- started manually with:   bash bench/run_night.sh
#
# Data: data_f30a14d/ -- the other/ features relabelled with LSTM_HRC commit
# f30a14d (bench/relabel.py). Lift is TRIMMED to the part before Place/Align and
# kept as a class; idle ("No Related Task") = frames no annotation covers.
# other/ had removed Lift entirely, so 41.6% of the real lifting frames were
# labelled idle -- round 1 trained on that. Because the labels changed, round 1's
# N1 is not a fair baseline for these arms; B0 re-establishes it.
#
#   B0  baseline: all 16 panels, idle -> bg head, 30 fps, 4 s      (paired anchor)
#   F1  data-driven 7 panels (bench/feat_importance.py), 121 dims
#   C1  Screw and Clamp Coupling weighted 2x in the loss
#   S1  trained at 10 fps (stride 3), 4 s window = 40 samples -- matches the live loop
#   S2  10 fps, 8 s window = 80 samples
#   S3  S2 + idle-head loss weight 0.2 -> 0.5
#   L1  LSTM with B0's settings (runtime and team code use AssistLSTM)
# Roughly 4.5 hours in total.
set -e
cd "$(dirname "$0")/.."
PY=hrc_communication/.venv/Scripts/python.exe
D=data_f30a14d
BASE="--targets peak --aug mirror --data $D/original --aug-dir $D/augmented_mirror --keep-lift --save-models"

$PY bench/loso.py $BASE --panels all                                              --tag B0_f30a_base      > bench/logs/B0.log 2>&1
$PY bench/loso.py $BASE --panels selected                                         --tag F1_selected       > bench/logs/F1.log 2>&1
$PY bench/loso.py $BASE --panels all --class-weight "Screw=2,Clamp Coupling=2"    --tag C1_screw_clamp    > bench/logs/C1.log 2>&1
$PY bench/loso.py $BASE --panels all --stride 3 --win 40                          --tag S1_10fps_4s       > bench/logs/S1.log 2>&1
$PY bench/loso.py $BASE --panels all --stride 3 --win 80 --epochs 10              --tag S2_10fps_8s       > bench/logs/S2.log 2>&1
$PY bench/loso.py $BASE --panels all --stride 3 --win 80 --epochs 10 --bg-weight 0.5 --tag S3_10fps_8s_bg05 > bench/logs/S3.log 2>&1
$PY bench/loso.py $BASE --panels all --model lstm                                 --tag L1_lstm           > bench/logs/L1.log 2>&1
echo NIGHT_DONE

# Export. Rule fixed BEFORE seeing results: the live loop runs at 10 Hz, so only
# the 10 fps arms (S1-S3) are deployable; ship whichever has the highest LOSO
# mean macro-F1, retrained on all 15 subjects (bench/deploy.py).
BEST=$($PY -c "
import json
s={t: json.load(open('bench/results/loso_%s.json'%t))['summary']['macro_f1_mean']
   for t in ['S1_10fps_4s','S2_10fps_8s','S3_10fps_8s_bg05']}
print(max(s, key=s.get))")
echo "deploying $BEST"
$PY bench/deploy.py --from-run "$BEST" --out-dir "models/$BEST" > bench/logs/deploy.log 2>&1
echo DEPLOY_DONE
