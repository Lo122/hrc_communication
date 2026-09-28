set -e
PY=hrc_communication/.venv/Scripts/python.exe
# Wait for the 2x2 queue. pgrep cannot see Windows processes from Git Bash, so
# poll the expected result files instead -- the queue writes one per arm.
while [ "$(ls bench/results/loso_plateau_aug.json bench/results/loso_peak_noaug.json bench/results/loso_peak_aug.json 2>/dev/null | wc -l)" -lt 3 ]; do
  sleep 60
done
$PY bench/loso.py --targets vector --aug none --tag plateau_noaug --save-models \
    > bench/logs/plateau_noaug_folds.log 2>&1
$PY bench/trigger_eval.py --arms plateau_noaug > bench/logs/trigger_eval.log 2>&1
echo TRIGGER_DONE
