MODEL="$1"
TEST_PRE="/mnt/disks/disk-3/mikawa/asj/data/ASJ_TEST_VALIDS/no_ovlp/no_ovlp.json"
TEST_P10="/mnt/disks/disk-3/mikawa/asj/data/ASJ_TEST_VALIDS/snr10/snr10.json"
TEST_P05="/mnt/disks/disk-3/mikawa/asj/data/ASJ_TEST_VALIDS/snr05/snr05.json"
TEST_P00="/mnt/disks/disk-3/mikawa/asj/data/ASJ_TEST_VALIDS/snr00/snr00.json"
TEST_N05="/mnt/disks/disk-3/mikawa/asj/data/ASJ_TEST_VALIDS/rns05/rns05.json"
TEST_N10="/mnt/disks/disk-3/mikawa/asj/data/ASJ_TEST_VALIDS/rns10/rns10.json"
# TEST_RND="/mnt/disks/disk-3/mikawa/asj/data/ASJ_TEST_VALIDS/randm/randm.json"
TEST_RND="/mnt/t4/nemo/ASJ_FINETUNE_SET/ASJ_TEST_VALIDS/randm/randm.json"

EXP_DIR="$2"

BATCH_SIZE=8

echo $MODEL

# python speech_to_text_eval.py \
#     model_path=$MODEL \
#     dataset_manifest=$TEST_PRE \
#     result_path="$EXP_DIR/result_pretr.txt" \
#     output_filename="$EXP_DIR/predictions_pretr.json" \
#     batch_size=$BATCH_SIZE \
#     use_cer=True \

# python speech_to_text_eval.py \
#     model_path=$MODEL \
#     dataset_manifest=$TEST_P10 \
#     result_path="$EXP_DIR/result_snr10.txt" \
#     output_filename="$EXP_DIR/predictions_snr10.json" \
#     batch_size=$BATCH_SIZE \
#     use_cer=True \

# python speech_to_text_eval.py \
#     model_path=$MODEL \
#     dataset_manifest=$TEST_P05 \
#     result_path="$EXP_DIR/result_snr05.txt" \
#     output_filename="$EXP_DIR/predictions_snr05.json" \
#     batch_size=$BATCH_SIZE \
#     use_cer=True \

# python speech_to_text_eval.py \
#     model_path=$MODEL \
#     dataset_manifest=$TEST_P00 \
#     result_path="$EXP_DIR/result_snr00.txt" \
#     output_filename="$EXP_DIR/predictions_snr00.json" \
#     batch_size=$BATCH_SIZE \
#     use_cer=True \

# python speech_to_text_eval.py \
#     model_path=$MODEL \
#     dataset_manifest=$TEST_N05 \
#     result_path="$EXP_DIR/result_rns05.txt" \
#     output_filename="$EXP_DIR/predictions_rns05.json" \
#     batch_size=$BATCH_SIZE \
#     use_cer=True \

# python speech_to_text_eval.py \
#     model_path=$MODEL \
#     dataset_manifest=$TEST_N10 \
#     result_path="$EXP_DIR/result_rns10.txt" \
#     output_filename="$EXP_DIR/predictions_rns10.json" \
#     batch_size=$BATCH_SIZE \
#     use_cer=True \

python speech_to_text_eval.py \
    model_path=$MODEL \
    dataset_manifest=$TEST_RND \
    result_path="$EXP_DIR/result_randm.txt" \
    output_filename="$EXP_DIR/predictions_randm.json" \
    batch_size=$BATCH_SIZE \
    use_cer=True \