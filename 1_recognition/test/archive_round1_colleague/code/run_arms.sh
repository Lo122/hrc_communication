set -e
PY=hrc_communication/.venv/Scripts/python.exe
mkdir -p bench/results bench/logs
# 2x2: {plateau, peak} x {no aug, mirror aug}.  GRU throughout -- the measured
# model gap (0.026) is inside the split-noise floor (0.085), so spending 4x the
# compute on a second architecture would resolve a difference that isn't there.
$PY bench/loso.py --targets vector --aug none   --tag plateau_noaug > bench/logs/plateau_noaug.log 2>&1
$PY bench/loso.py --targets vector --aug mirror --tag plateau_aug   > bench/logs/plateau_aug.log   2>&1
$PY bench/loso.py --targets peak   --aug none   --tag peak_noaug    > bench/logs/peak_noaug.log    2>&1
$PY bench/loso.py --targets peak   --aug mirror --tag peak_aug      > bench/logs/peak_aug.log      2>&1
echo "ALL 4 ARMS DONE"
