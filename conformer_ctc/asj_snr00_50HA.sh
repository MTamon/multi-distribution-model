TRAIN_MANIFEST=/mnt/t4/nemo/ASJ_SNR00_50HA/manifest/train_nodup_sp/train_nodup_sp_manifest.json
VALID_MANIFEST=/mnt/t4/nemo/ASJ_SNR00_50HA/manifest/valid/valid_manifest.json
TESTS_MANIFEST=/mnt/t4/nemo/ASJ_SNR00_50HA/manifest/train_dev/train_dev_manifest.json
VOCAB_PATH=./asj_exp/pretrain/vocab_CTC.txt
NEMO_PATH=./models/confCTC_BEST.nemo
EXP_DIR=./asj_exp/snr00_50ha
NAME="snr00_50ha"

# LR=0.058977
# LR=0.066
LR=0.1143
MIN_LR=1e-7
WARMUP_STEP=3000
BATCH_SIZE=32
GA=4

python speech_to_text_ctc.py \
    --config-path=./ --config-name=conformer_ctc_char.yaml \
    name="$NAME" \
    model.train_ds.manifest_filepath="$TRAIN_MANIFEST" \
    model.validation_ds.manifest_filepath="$VALID_MANIFEST" \
    model.test_ds.manifest_filepath="$TESTS_MANIFEST" \
    trainer.accelerator='gpu' \
    trainer.max_epochs=200 \
    mode="Pretrain" \
    exp_dir="$EXP_DIR" \
    mode="Finetune" \
    nemo_path="$NEMO_PATH" \
    exp_manager.wandb_logger_kwargs.project="asj_overlap_ctc" \
    exp_manager.checkpoint_callback_params.save_top_k=10 \
    vocab="$VOCAB_PATH" \
    lr=$LR \
    model.optim.sched.min_lr=$MIN_LR \
    warmup_steps=$WARMUP_STEP \
    batch_size=$BATCH_SIZE \
    accumulate_grad_batches=$GA \
    trainer.precision=16 \
    num_workers=16
