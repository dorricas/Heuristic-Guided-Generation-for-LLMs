#!/bin/bash
CUDA_VISIBLE_DEVICES=0
export llama_path="/home/sumedh/meta-llama"  # or "your/path/to/llama2"
export llama_size="1B"                       # or "7B" of llama2
export base_lm=hf                          # or "llama2"
# export model_dir="Qwen/Qwen2-7B-Instruct"
export model_dir="meta-llama/Llama-3.2-3B"

temps=(0.1 1 2 3 4)
time_budgets=(4 8 12 16 20)
for t in "${temps[@]}"; do
	for budget in "${time_budgets[@]}"; do
	    python -m torch.distributed.run --nproc_per_node=1 examples/ToT/prontoqa/tot_inference.py \
		--depth_limit 10 \
		--base_lm "$base_lm" \
		--model_dir "$model_dir" \
		--temperature 0.8 \
		--search_algo lts \
		--total_calls 10000 \
		--max_time "$budget" \
		--max_per_state 3 \
		--lts_temp "$t" \
		--llama_size "$llama_size" \
		--log_dir "logs/lts_prontoqa_${budget}_${t}" | tee "outputs/prontoqa_lts_${budget}_${t}"
	done
done
