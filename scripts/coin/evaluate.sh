model=online_lmm
llama_pretrained=meta-llama/Llama-2-7b-chat-hf
vision_pretrained=laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
connector_pretrained=mlp
eval_datasets="coin_step_test coin_next_test coin_task_test coin_procedure_test coin_taskprocedure_test"
stream_loss_weight=0.0
resume_from_checkpoint="outputs/online_lmm-laion--CLIP-ViT-L-14-DataComp.XL-s13B-b90K-meta-llama--Llama-2-7b-chat-hf/coin_benchmarks/bs64_lr0.0001_10e_^model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|lm_head\$_stream"

if [ -n "$MASTER_ADDR" ]; then
    launcher="torchrun --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
else
    launcher="torchrun --nproc_per_node 8"
fi

$launcher evaluate.py $model \
    --eval_datasets $eval_datasets \
    --llama_pretrained $llama_pretrained --vision_pretrained $vision_pretrained --connector_pretrained $connector_pretrained \
    --load_vision_embeds True \
    --stream_loss_weight $stream_loss_weight \
    --resume_from_checkpoint $resume_from_checkpoint \
    --output_dir $resume_from_checkpoint \
    --per_device_eval_batch_size 1 \
    --prediction_loss_only False \
    --dataloader_num_workers 32 \
    --fp16_full_eval True \
    --report_to tensorboard
