set -e
PY=hrc_communication/.venv/Scripts/python.exe
D=hrc_communication/other
BASE="--targets peak --aug mirror --data $D/original --aug-dir $D/augmented_mirror --save-models"

# N1 (16 panels, idle -> bg head) is already running in its own process.
# It is kept as the paired anchor for W1/W2/L1 even though the annotation team
# does not need it for their question. N2 was dropped at their request.
while [ ! -f bench/results/loso_N1_full_bg.json ]; do sleep 60; done

# ROUND 1 -- annotation team's question: which feature subset?
$PY bench/loso.py $BASE --panels reduced                      --tag N3_red_bg     > bench/logs/N3.log 2>&1
$PY bench/loso.py $BASE --panels reduced     --idle-as-class  --tag N4_red_idle   > bench/logs/N4.log 2>&1
$PY bench/loso.py $BASE --panels reduced_vel                  --tag N5_redvel_bg  > bench/logs/N5.log 2>&1
$PY bench/loso.py $BASE --panels reduced_vel --idle-as-class  --tag N6_redvel_idle > bench/logs/N6.log 2>&1
echo ROUND1_DONE

# ROUND 2 -- measured improvements, compared against N1.
# Epochs lowered to 10: validation peaks ~epoch 6 and declines by 12.
$PY bench/loso.py $BASE --panels all --win 240 --epochs 10                  --tag W1_win240      > bench/logs/W1.log 2>&1
$PY bench/loso.py $BASE --panels all --win 240 --epochs 10 --bg-weight 0.5  --tag W2_win240_bg05 > bench/logs/W2.log 2>&1
echo ROUND2_DONE
# ROUND 3 (L1, LSTM) is started by run_round3.sh once both W files exist.
