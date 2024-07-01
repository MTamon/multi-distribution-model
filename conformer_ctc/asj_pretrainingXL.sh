TRAIN_MANIFEST=/mnt/t4/nemo/ASJ_PRETRAINED/manifest/train_nodup_sp/train_nodup_sp_manifest.json
VALID_MANIFEST=/mnt/t4/nemo/ASJ_PRETRAINED/manifest/valid/valid_manifest.json
TESTS_MANIFEST=/mnt/t4/nemo/ASJ_PRETRAINED/manifest/train_dev/train_dev_manifest.json
VOCAB_PATH=./asj_exp/pretrain/vocab_CTC.txt
NEMO_PATH=./models/confCTC_BEST.nemo
EXP_DIR=./asj_exp/pretrain
NAME="cradle-04-D1A"

python conformer_ctc/speech_to_text_ctc.py \
    --config-path=./ --config-name=conformer_ctc_charXL.yaml \
    name="$NAME" \
    model.train_ds.manifest_filepath="$TRAIN_MANIFEST" \
    model.validation_ds.manifest_filepath="$VALID_MANIFEST" \
    model.test_ds.manifest_filepath="$TESTS_MANIFEST" \
    trainer.accelerator='gpu' \
    trainer.max_epochs=100 \
    mode="Pretrain" \
    exp_dir="$EXP_DIR" \
    # mode="Finetune" \
    # vocab="$VOCAB_PATH" \
    # nemo_path="$NEMO_PATH" \