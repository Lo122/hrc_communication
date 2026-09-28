set -e
PY=hrc_communication/.venv/Scripts/python.exe
D=hrc_communication/other
# Wait for round 2 so nothing shares the GPU.
while [ "$(ls bench/results/loso_W1_win240.json bench/results/loso_W2_win240_bg05.json \
            2>/dev/null | wc -l)" -lt 2 ]; do sleep 60; done
# GRU vs LSTM was only ever compared on a single split, where the 0.021 gap was
# a quarter of the measured split-to-split noise. This pairs LSTM against N1
# (identical settings, GRU) on the same 15 folds. It also matters for
# deployment: the runtime and the annotation team's code both use AssistLSTM.
$PY bench/loso.py --model lstm --targets peak --aug mirror --data $D/original \
    --aug-dir $D/augmented_mirror --save-models --panels all \
    --tag L1_lstm > bench/logs/L1.log 2>&1
echo ROUND3_DONE
