LOGDIR=/home/jiaoyang/code/ScanRefer/log

python scripts/train.py \
    --use_color \
    | tee ${LOGDIR}/"$(date)".log