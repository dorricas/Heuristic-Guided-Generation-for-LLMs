CUDA_VISIBLE_DEVICES=0
export llama_path="/home/sumedh/meta-llama/" # or "your/path/to/llama2"
export llama_size="1B" # or "7B" of llama2
export base_lm="$1" # or "llama2"
export model_dir="Qwen/Qwen2-7B-Instruct"
# export model_dir="meta-llama/Llama-3.2-3B"

python -m torch.distributed.run --nproc_per_node 1 examples/ToT/prontoqa/tot_inference.py --depth_limit 10 --base_lm hf --model_dir $model_dir --temperature 0.8 --search_algo lts --total_calls 100 --max_per_state 3 --llama_size $llama_size --log_dir logs/lts_prontoqa
