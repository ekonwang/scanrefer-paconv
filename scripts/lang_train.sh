LOGDIR=/home/jiaoyang/code/ScanRefer/log

python scripts/train.py \
    --use_color \
    --use_lang_paconv \
    --use_paconv \
    | tee ${LOGDIR}/"$(date)".log