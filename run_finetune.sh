export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1
torchrun --nproc_per_node 2 \
-m FlagEmbedding.llm_reranker.finetune_for_instruction.run \
--output_dir save_path/init_selftrain_flashattn \
--model_name_or_path ./gemma-2b \
--train_data finetune_data_bge_jsonl/selftrain_full.jsonl \
--learning_rate 2e-4 \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--dataloader_drop_last True \
--query_max_len 128 \
--passage_max_len 512 \
--train_group_size 16 \
--logging_steps 1 \
--save_steps 2000 \
--save_total_limit 50 \
--ddp_find_unused_parameters False \
--gradient_checkpointing \
--deepspeed ds_config.json \
--warmup_ratio 0.1 \
--bf16 \
--use_lora True \
--lora_rank 16 \
--lora_alpha 32 \
--use_flash_attn False \
--target_modules q_proj k_proj v_proj o_proj