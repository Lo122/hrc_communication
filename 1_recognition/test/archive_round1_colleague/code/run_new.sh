set -e
PY=hrc_communication/.venv/Scripts/python.exe
D=hrc_communication/other
mkdir -p bench/results bench/logs
# New corpus: Lift already emptied by the annotators, "No Related Task"
# annotated explicitly (18.2% of frames). Two questions, crossed:
#   idle as bg head vs idle as a 7th trainable class
#   all 16 feature panels (251 dims) vs the 6 requested panels (89 dims)
BASE="--targets peak --aug mirror --data $D/original --aug-dir $D/augmented_mirror --save-models"
$PY bench/loso.py $BASE --panels all                       --tag N1_full_bg    > bench/logs/N1.log 2>&1
$PY bench/loso.py $BASE --panels all     --idle-as-class   --tag N2_full_idle  > bench/logs/N2.log 2>&1
$PY bench/loso.py $BASE --panels reduced                   --tag N3_red_bg     > bench/logs/N3.log 2>&1
$PY bench/loso.py $BASE --panels reduced --idle-as-class   --tag N4_red_idle   > bench/logs/N4.log 2>&1
echo ALL_NEW_ARMS_DONE
