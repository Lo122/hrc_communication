set -e
PY=hrc_communication/.venv/Scripts/python.exe
D=hrc_communication/other
BASE="--targets peak --aug mirror --data $D/original --aug-dir $D/augmented_mirror --save-models"
# N1 is still running in its own process; start after it so the GPU is not shared.
while [ ! -f bench/results/loso_N1_full_bg.json ]; do sleep 60; done
$PY bench/loso.py $BASE --panels reduced                      --tag N3_red_bg      > bench/logs/N3.log 2>&1
$PY bench/loso.py $BASE --panels reduced     --idle-as-class  --tag N4_red_idle    > bench/logs/N4.log 2>&1
$PY bench/loso.py $BASE --panels reduced_vel                  --tag N5_redvel_bg   > bench/logs/N5.log 2>&1
$PY bench/loso.py $BASE --panels reduced_vel --idle-as-class  --tag N6_redvel_idle > bench/logs/N6.log 2>&1
echo ROUND1_DONE
