CUDA_VISIBLE_DEVICES=0
export llama_path="/home/sumedh/meta-llama/" # or "your/path/to/llama2"
export llama_size="1B" # or "7B" of llama2
export base_lm="$1" # or "llama2"

export PYTHONPATH="${PYTHONPATH}:/home/sumedh/Search-LLM-inference"

python -m torch.distributed.run --nproc_per_node 1 examples/ToT/sorting/tot_inference_sort.py --depth_limit 10 --base_lm hf --model_dir meta-llama/Llama-3.1-8B-Instruct --temperature 0.8 --search_algo dfs --total_calls 1000 --max_per_state 3 --llama_size $llama_size --log_dir logs/dfs_sort
