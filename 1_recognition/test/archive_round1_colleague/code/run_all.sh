set -e
PY=hrc_communication/.venv/Scripts/python.exe
mkdir -p bench/results
for m in lstm gru tcn lstm2; do
  echo "########## $m (balanced) ##########"
  $PY bench/train.py --model $m --epochs 14 --hop 10 --balanced \
      --out bench/results/${m}_bal.json 2>&1 | grep -v UserWarning | grep -v "tot +="
done
echo "########## lstm WITHOUT class weights (ablation) ##########"
$PY bench/train.py --model lstm --epochs 14 --hop 10 \
    --out bench/results/lstm_unbal.json 2>&1 | grep -v UserWarning | grep -v "tot +="
