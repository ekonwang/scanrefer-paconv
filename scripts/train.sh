LOGDIR=/home/jiaoyang/code/ScanRefer/log

python scripts/train.py \
    --use_color \
    --use_paconv \
    --gpu 1 \
    | tee ${LOGDIR}/"$(date)".log