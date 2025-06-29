SEED=0

uv run src/ggdg/main.py \
    --seed $SEED \
    --exp_name gdg/Qwen2.5-1.5B-Instruct_SFT_GRPO/${SEED} \
    --dataset ludii \
    --engine local/log/models/grpo/Qwen2.5-1.5B-Instruct_SFT_GRPO \
    --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
    --output_max_tokens 2048 \
    --input_max_tokens 8192 \
    --prompt_mode std \
    --game_list_path "data/ludii/eval/default.json" \
    --test_num 100 \
    --num_shot 0 \
    --prompt_template_path prompts/grpo.json \
