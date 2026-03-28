import pickle
from typing import Type, Optional, Literal

import numpy as np
from tqdm import tqdm
from datetime import datetime
import os, sys

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import BeamSearch, LTS, DFS , MCTSNode
from reasoners.visualization import TreeLog

from world_model import Game24WorldModel, Game24State, Game24Action
from search_config import Game24Config
import utils


def node_visualizer(x: MCTSNode):
    ret = {}
    if x.action is not None:
        ret['last_step'] = x.action
    if x.state is not None:
        ret = {'current': x.state.current}
        if x.state.output is not None:
            ret['output'] = x.state.output
    return ret

def tot_game24(base_model: LanguageModel,
           prompts: dict,
           calc_reward: Literal['sampling', 'logits'] = 'logits',
           search_algo: str = "beam",
           n_action: int = 4,
           n_eval: int = 3,
           batch_size: int = 3,
           resume: int = 0,
           depth_limit: int = 6,
           log_dir: Optional[str] = None,
           disable_log: bool = False,
           temperature: float = 0.8,
           n_beam: int = 5,
           **search_algo_params):

    if search_algo == "beam":
        search_algo_params |= {"max_depth": depth_limit, 'beam_size':n_beam}
        search_algo = BeamSearch(**search_algo_params)
    elif search_algo == "dfs":
        search_algo_params |= {"depth": depth_limit}
        search_algo = DFS(**search_algo_params)
    elif search_algo == "lts":
        search_algo = LTS(**search_algo_params)
        pass
        #search_algo_params |= {"depth": depth_limit}
    else:
        print("Unknown search algorithm", search_algo)
        raise NotImplementedError
    
    world_model = Game24WorldModel(base_model=base_model, prompt=prompts, batch_size=batch_size)
    config = Game24Config(base_model=base_model, prompt=prompts, calc_reward=calc_reward, temperature=temperature,
                          n_actions=n_action, n_eval=n_eval, batch_size=batch_size, depth_limit=depth_limit,)

    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    # test from 900-999
    dataset = utils.read_data(file='./examples/ToT/game24/data/24.csv')[900:1000]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir,'algo_output'), exist_ok=True)
    correct_count = 0
    for i, example in enumerate(tqdm(dataset, total=len(dataset), initial=0, desc='game24')):
        # print(f'\n======== example {i}: {example} ========')
        reasoner.world_model = Game24WorldModel(base_model=base_model, prompt=prompts, batch_size=batch_size)
        # algo_output = reasoner(example, action_dedup=True, return_beam=True, early_terminate=False,
        #                        reward_strategy='last_iter')
        algo_output = reasoner(example)
        output = algo_output.terminal_state.output if algo_output.terminal_state is not None else None
        # print(output)
        correct = utils.test_output(example, output)

        correct_count += correct
        accuracy = correct_count / (i + 1)
        log_str = f'Case #{resume + i + 1}: {correct=}, {output=} ; {accuracy=:.3f} ({correct_count}/{i + 1})'
        tqdm.write(log_str)
        if not disable_log:
            with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                print(log_str, file=f)
            with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.pkl'), 'wb') as f:
                pickle.dump(algo_output, f)
            """
            if isinstance(search_algo, MCTS):
                with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.json'), 'w') as f:
                    # noinspection PyTypeChecker
                    print(TreeLog.from_mcts_results(algo_output, node_data_factory=node_visualizer), file=f)
            """
    print("Accuracy:",accuracy)


if __name__ == '__main__':
    import os
    import sys
    import json
    import warnings
    import fire
    import random

    llama_ckpts = os.environ.get("LLAMA_CKPTS", None)
    llama_2_ckpts = os.environ.get("LLAMA_2_CKPTS", None)
    llama_3_ckpts = os.environ.get("LLAMA3_CKPTS", None)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        sys.stdout = open(os.devnull, 'w')
        warnings.filterwarnings('ignore')

    def main(base_lm: Literal['llama', 'llama.cpp', 'llama2', 'hf', 'exllama','llama3','openai'] = 'llama3',
             temperature = 0.8,
             llama_ckpts: str = llama_ckpts,
             model_dir = '/path/to/model',
             llama_size: str = '13B',
             llama_cpp_path: str = None,
             llama_cpp_n_batch: int = 512,
             hf_path: str = 'meta-llama/Llama-2-13b-hf',
             hf_peft_path: Optional[str] = None,
             hf_quantized: Optional[Literal['awq', 'int8', 'fp4', 'nf4']] = None,
             hf_load_awq_path: Optional[str] = None,
             exllama_model_dir: str = 'WizardMath-13B-V1.0-GPTQ',
             exllama_lora_dir: Optional[str] = None,
             exllama_mem_map: Optional[str] = None,
             batch_size: int = 1,
             search_algo = "beam",
             prompts: str = 'examples/ToT/game24/prompts/game24.json',
             disable_log: bool = False,
             disable_tqdm: bool = False,
             **kwargs):
        with open(prompts) as f:
            prompts = json.load(f)
        if base_lm in ['llama', 'llama2', 'llama3', 'llama3.2']:
            import torch
            import torch.backends.cudnn
            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True
        
        llama_ckpts = model_dir
        llama_2_ckpts = model_dir
        llama_3_ckpts = model_dir
        
        if base_lm == 'llama':
            from reasoners.lm import LlamaModel
            base_model = LlamaModel(llama_ckpts, llama_size, max_batch_size=batch_size)
        elif base_lm == 'llama.cpp':
            from reasoners.lm import LlamaCppModel
            base_model = LlamaCppModel(llama_cpp_path, n_batch=llama_cpp_n_batch)
        elif base_lm == 'llama2':
            from reasoners.lm import Llama2Model
            base_model = Llama2Model(llama_2_ckpts, llama_size, max_batch_size=batch_size)
        elif base_lm == 'llama3' or base_lm == 'llama3.2':
            from reasoners.lm import Llama3Model
            base_model = Llama3Model(llama_3_ckpts, llama_size, max_batch_size=batch_size)
        elif base_lm == 'hf':
            from reasoners.lm import HFModel
            model = HFModel(model_dir, model_dir, max_batch_size=batch_size)
        elif base_lm == 'openai':
            from reasoners.lm import OpenAIModel
            base_model = OpenAIModel(model='gpt-3.5-turbo', temperature=temperature, max_tokens=2048 )
        elif base_lm == 'exllama':
            from reasoners.lm import ExLlamaModel
            base_model = ExLlamaModel(exllama_model_dir, exllama_lora_dir, mem_map=exllama_mem_map,
                                      max_batch_size=batch_size, max_new_tokens=512, max_seq_length=2048)
        else:
            assert False, f'cannot resolve {base_lm=}'
        tot_game24(base_model=base_model,
                   prompts=prompts,
                   batch_size=batch_size,
                   n_beam=5,
                   disable_log=disable_log or local_rank != 0,
                   search_algo=search_algo,
                   **kwargs)

    fire.Fire(main)
