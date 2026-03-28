CUDA_VISIBLE_DEVICES=0
export llama_path="/home/sumedh/meta-llama/" # or "your/path/to/llama2"
export llama_size="1B" # or "7B" of llama2
export base_lm="$1" # or "llama2"


python -m torch.distributed.run --nproc_per_node 1 examples/ToT/prontoqa/tot_inference.py --depth_limit 10 --base_lm $base_lm --model_dir $llama_path --temperature 0.8 --search_algo dfs --total_states 10 --max_per_state 3 --llama_size $llama_size --log_dir logs/dfs_prontoqa
