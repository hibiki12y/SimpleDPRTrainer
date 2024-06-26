torchrun --nproc_per_node 1 main.py \
        --output_dir checkpoints/roberta_dpr_testbed \
        --base_model_name_or_path roberta-base \
        --train_passage data/train_passages.jsonl \
        --train_query data/train_queries.jsonl \
        --dev_passage data/dev_passages.jsonl \
        --dev_query data/dev_queries.jsonl \
        --max_passage_len 512 \
        --max_query_len 128 \
        --per_device_train_batch_size=6 \
        --per_device_eval_batch_size=6 \
        --num_train_epochs=10 \
        --warmup_steps=1000 \
        --logging_steps=100 \
        --save_total_limit=1 \
        --evaluation_strategy=steps \
        --save_strategy=steps \
        --eval_steps=200 \
        --save_steps=200 \
        --remove_unused_columns False \
        --ddp_find_unused_parameters False \
        --learning_rate 5e-5 \
        --report_to tensorboard \
        --optim "adamw_hf" \
        --bf16 \
        --torch_compile
