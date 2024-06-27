export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1000000,garbage_collection_threshold:0.3'

python asj_conformerT.py \
    --ckpt-name confT_pretrain.nemo \
    --ckpt-dir models \
    --train-path /mnt/t4/nemo/ASJ_PRETRAINED/manifest/train_nodup_sp/train_nodup_sp_manifest.json \
    --test-path /mnt/t4/nemo/ASJ_PRETRAINED/manifest/train_dev/train_dev_manifest.json \
    --valid-path /mnt/t4/nemo/ASJ_PRETRAINED/manifest/valid/valid_manifest.json \
    --scheduler NoamAnnealing \
    --lr 3.0 \
    --min-lr 1e-6 \
    --max-epoch 1 \
    --warmup-steps 6600 \
    --constant-epochs 1 \
    --weight-decay 0.001 \
    --ga 16 \
    --batch-size 8 \
    --epochs 100 \
    --top-k 5 \
    --epoch-logging-n 100 \
    --data-path /mnt/t4/nemo/ASJ_PRETRAINED \
    --yaml-path /home/mikawa/empath/intern/NeMo/transducer_cf/hparams.yaml \
    --train-dir-name train \
    --valid-dir-name valid \
    --proj-name asj \
    --devices 1 \
    --accelerator gpu \
    --strategy ddp
