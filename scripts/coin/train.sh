if [ -n "$MASTER_ADDR" ]; then
    launcher="torchrun --nproc_per_node 8 --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
    nnodes=$SLURM_NNODES
else
    launcher="torchrun --nproc_per_node 8"
    nnodes=1
fi

model=online_lmm
lora_r=128
lora_alpha=256
lora_modules="^model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|lm_head$"
finetune_modules="connector"
per_device_train_batch_size=2
gradient_accumulation_steps=$((8/$per_device_train_batch_size/$nnodes))
learning_rate=0.0001
llama_pretrained=meta-llama/Llama-2-7b-chat-hf
vision_pretrained=laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
connector_pretrained=mlp
train_datasets="coin_step_train coin_next_train coin_task_train coin_procedure_train coin_taskprocedure_train"
eval_datasets="coin_step_test coin_next_test coin_task_test coin_procedure_test coin_taskprocedure_test"
load_vision_embeds=True
epoch=10
output_dir="outputs/${model}-${vision_pretrained//\//--}-${llama_pretrained//\//--}/coin_benchmarks/bs64_lr${learning_rate}_${epoch}e_${lora_modules}_stream${stream_loss_weight}"

${launcher} train.py ${model} \
    --deepspeed configs/deepspeed/zero1.json \
    --llama_pretrained $llama_pretrained \
    --vision_pretrained $vision_pretrained \
    --connector_pretrained $connector_pretrained \
    --lora_modules $lora_modules --lora_r $lora_r --lora_alpha $lora_alpha \
    --finetune_modules $finetune_modules \
    --train_datasets $train_datasets \
    --eval_datasets $eval_datasets \
    --load_vision_embeds $load_vision_embeds \
    --output_dir $output_dir \
    --num_train_epochs $epoch \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --gradient_checkpointing True \
    --evaluation_strategy no \
    --prediction_loss_only False \
    --save_strategy no \
    --learning_rate ${learning_rate} \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --logging_steps 10 \
    --dataloader_num_workers 32 \
    --bf16 True \
    --tf32 True \
    --report_to tensorboard \
