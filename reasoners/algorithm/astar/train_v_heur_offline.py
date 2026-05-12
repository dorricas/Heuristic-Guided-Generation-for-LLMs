import os
import sys
import copy
import json
import random
import datetime
import time
from collections import deque
import logging

import torch
import fire
from peft import get_peft_model, LoraConfig, TaskType

# Ensure we can import from the main project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from examples.ToT.blocksworld.tot_inference import BWConfig, BlocksWorldModel, BWState, dfs_bw_extractor
from reasoners.benchmark.blocksworld import BWEvaluator
import reasoners.benchmark.bw_utils as bw_utils

from reasoners.algorithm.astar.train_v_heur import BWThoughtEnv, ThoughtHeurModel, HeuristicTrainer

# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
def main(
    buffer_path: str,
    model_dir: str = '',
    prompt_path: str = 'examples/CoT/blocksworld/prompts/pool_prompt_v1.json',
    data_path: str = 'examples/CoT/blocksworld/data/split_v1/split_v1_step_2_data.json',
    config_file: str = 'examples/CoT/blocksworld/data/bw_config.yaml',
    domain_file: str = 'examples/CoT/blocksworld/data/generated_domain.pddl',
    output_dir: str = 'logs/train_v_heur_offline',
    batch_size: int = 16,
    max_steps: int = 5000,
    log_every: int = 50,
    checkpoint_every: int = None,
    resume: str = None
):
    # ---------------------------------------------
    # SETUP LOGGING AND DIRS
    # ---------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Setting up OFFLINE training environment... Results will be saved to: {run_dir}")
    print(f"Loading replay buffers from: {buffer_path}")

    # ---------------------------------------------
    # LOAD PROMPT DICT
    # ---------------------------------------------
    with open(prompt_path) as f:
        prompt = json.load(f)

    # ---------------------------------------------
    # SETUP ToT BLOCKSWORLD INFRASTRUCTURE & LLM
    # ---------------------------------------------
    if "VAL" not in os.environ:
        val_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../LLMs-Planning/planner_tools/VAL'))
        if os.path.exists(val_path):
            os.environ["VAL"] = val_path
        else:
            print(f"WARNING: The VAL environment variable is missing and could not be found at {val_path}.")
            
    from reasoners.lm.hf_model import HFModel
    
    if not hasattr(torch.nn.Module, "set_submodule"):
        def _set_submodule(self, target: str, module: torch.nn.Module) -> None:
            atoms = target.split(".")
            name = atoms.pop(-1)
            mod = self
            for item in atoms:
                mod = getattr(mod, item)
            setattr(mod, name, module)
        torch.nn.Module.set_submodule = _set_submodule
    
    print(f"Loading {model_dir} in 4-bit NF4 for PEFT LoRA training...")
    hf_wrapper = HFModel(model_dir, model_dir, device="cuda:0", 
                        max_batch_size=batch_size, max_new_tokens=200, 
                        quantized="nf4")
    
    base_llm = hf_wrapper.model
    tokenizer = hf_wrapper.tokenizer
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Injecting LoRA Training Adapters using PEFT...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    peft_model = get_peft_model(base_llm, lora_config)
    peft_model.print_trainable_parameters()
    
    hf_wrapper.model = peft_model
    model = hf_wrapper
        
    world_model = BlocksWorldModel(base_model=model, prompt=prompt, max_steps=10)
    config = BWConfig(base_model=model, prompt=prompt, temperature=1.0, n_candidate=4)
    
    evaluator = BWEvaluator(config_file=config_file, domain_file=domain_file, data_path=data_path, 
                            init_prompt=prompt, output_extractor=dfs_bw_extractor)

    # ---------------------------------------------
    # INITIALIZE RL ENVIRONMENT AND HEUR TRAINER
    # ---------------------------------------------
    env = BWThoughtEnv(world_model, config, evaluator, use_cache=True)

    heur_model = ThoughtHeurModel(base_llm=peft_model, tokenizer=tokenizer, max_len=256)
    heur_model.to(peft_model.device)
    
    if resume:
        print(f"Resuming training from checkpoint: {resume}")
        heur_model.load_state_dict(torch.load(resume, map_location=peft_model.device), strict=False)
    
    trainer = HeuristicTrainer(env, heur_model)
    trainer.base_llm = peft_model 
    
    # ---------------------------------------------
    # LOAD REPLAY BUFFERS
    # ---------------------------------------------
    print(f"Loading replay buffers from {buffer_path}...")
    buffer_data = torch.load(buffer_path)
    
    # Override the deques so they don't drop elements if the loaded buffer is > 20000 or 30000
    pos_len = max(1, len(buffer_data['buffer_pos']))
    neg_len = max(1, len(buffer_data['buffer_neg']))
    trainer.buffer_pos = deque(buffer_data['buffer_pos'], maxlen=pos_len)
    trainer.buffer_neg = deque(buffer_data['buffer_neg'], maxlen=neg_len)
    
    print(f"Loaded {len(trainer.buffer_pos)} positive states and {len(trainer.buffer_neg)} negative states.")

    # ---------------------------------------------
    # OFFLINE TRAINING LOOP
    # ---------------------------------------------
    print(f"Beginning Offline Training for {max_steps} steps...")
    
    start_time = time.time()
    last_log_time = start_time
    
    history_loss = []
    history_heur_mean = []
    
    window_losses = []
    window_v_means = []
    window_v_stds = []
    window_v_mins = []
    window_v_maxs = []
    
    for step in range(1, max_steps + 1):
        res = trainer.train_step(batch_size=batch_size)
        if res:
            loss, (v_mean, v_std, v_min, v_max, grad_norm) = res
            window_losses.append(loss)
            window_v_means.append(v_mean)
            window_v_stds.append(v_std)
            window_v_mins.append(v_min)
            window_v_maxs.append(v_max)
            
            history_loss.append(loss)
            history_heur_mean.append(v_mean)
            
        if step % log_every == 0 and len(window_losses) > 0:
            current_time = time.time()
            elapsed_since_log = current_time - last_log_time
            
            avg_loss = sum(window_losses) / len(window_losses)
            avg_v_mean = sum(window_v_means) / len(window_v_means)
            avg_v_std = sum(window_v_stds) / len(window_v_stds)
            avg_v_min = sum(window_v_mins) / len(window_v_mins)
            avg_v_max = sum(window_v_maxs) / len(window_v_maxs)
            
            print(f"Step {step:05d}"
                  f" | Avg Loss: {avg_loss:.3f}"
                  f" | Avg Val: {avg_v_mean:.1f}"
                  f" | Avg Range: [{avg_v_min:.1f}-{avg_v_max:.1f}]"
                  f" | Avg Std: {avg_v_std:.2f}"
                  f" | Time/{log_every}Stp: {elapsed_since_log:.1f}s"
                  )
                  
            window_losses = []
            window_v_means = []
            window_v_stds = []
            window_v_mins = []
            window_v_maxs = []
            last_log_time = current_time

        if checkpoint_every is not None and step % checkpoint_every == 0:
            chkpt_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(chkpt_dir, exist_ok=True)
            chkpt_path = os.path.join(chkpt_dir, f"heur_model_step{step:05d}.pth")
            trainable_state_dict = {k: v for k, v in heur_model.state_dict().items() if "lora" in k or "value_head" in k}
            torch.save(trainable_state_dict, chkpt_path)
            print(f"  [>] Saved intermediate checkpoint to: {chkpt_path}")

    print(f"\nTraining Complete! Saving final Model Weights to {run_dir}/heur_model.pth")
    trainable_state_dict = {k: v for k, v in heur_model.state_dict().items() if "lora" in k or "value_head" in k}
    torch.save(trainable_state_dict, os.path.join(run_dir, "heur_model.pth"))
    
    # ---------------------------------------------
    # SAVE TRAINING GRAPHS
    # ---------------------------------------------
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        window = min(50, len(history_loss))
        if window > 1:
            smooth_loss = np.convolve(history_loss, np.ones(window)/window, mode='valid')
            smooth_heur = np.convolve(history_heur_mean, np.ones(window)/window, mode='valid')
            x_axis = range(window - 1, len(history_loss))
        else:
            smooth_loss = history_loss
            smooth_heur = history_heur_mean
            x_axis = range(len(history_loss))
        
        # 1. Loss Graph
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(history_loss)), history_loss, alpha=0.3, label="Raw Batch Loss", color="blue")
        plt.plot(x_axis, smooth_loss, label=f"Smoothed Loss (Window={window})", color="blue", linewidth=2.0)
        plt.xlabel("Updates")
        plt.ylabel("Loss")
        plt.title("Offline Neural Network Loss Over Time")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Heuristic Value Tracking Graph
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(history_heur_mean)), history_heur_mean, alpha=0.3, label="Raw Batch Mean Value", color="green")
        plt.plot(x_axis, smooth_heur, label=f"Smoothed Mean (Window={window})", color="green", linewidth=2.0)
        plt.xlabel("Updates")
        plt.ylabel("Avg H-Value Prediction")
        plt.title("Average Predicted Cost-To-Go over Time")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.savefig(os.path.join(run_dir, "value_curve.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved Training Logs and Graphs to {run_dir}")
    except ImportError:
        print("matplotlib not found, skipping graph generation.")

if __name__ == '__main__':
    fire.Fire(main)
