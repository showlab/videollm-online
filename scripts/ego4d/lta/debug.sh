export TOKENIZERS_PARALLELISM=false
if [ -n "$MASTER_ADDR" ]; then
    launcher="torchrun --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
    nnodes=$SLURM_NNODES
else
    launcher="torchrun --nproc_per_node 8"
    nnodes=1
fi

$launcher evaluate.py \
    --live_version live1+ \
    --eval_datasets ego4d_lta_test_unannotated \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --dataloader_num_workers 16 \
    --bf16_full_eval True \
    --report_to tensorboard \
    --output_dir outputs/debug \
    --resume_from_checkpoint outputs/ego4d_lta_train/live1+_2e-4_6e \
