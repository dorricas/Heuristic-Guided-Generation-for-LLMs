import pickle
from typing import Type, Callable, Optional, Literal

import numpy as np
from tqdm import tqdm
from datetime import datetime
import copy

from typing import NamedTuple
import reasoners.benchmark.bw_utils as utils

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners import WorldModel, LanguageModel, SearchConfig
from reasoners.benchmark import BWEvaluator
from reasoners.algorithm import BeamSearch, DFS, LTS

def bfs_bw_extractor(algo_output):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    # to make sure the plan is saved before evaluation in multi-process setting
    try:
        return "\n".join(algo_output.terminal_node.state.action_history)
    except Exception as e:
        print("Error in output extraction,", e)
        return ""
    
def dfs_bw_extractor(algo_output):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    # to make sure the plan is saved before evaluation in multi-process setting
    try:
        return "\n".join(algo_output.terminal_state.action_history)
    except Exception as e:
        print("Error in output extraction,", e)
        return ""

def lts_bw_extractor(algo_output):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    # to make sure the plan is saved before evaluation in multi-process setting
    try:
        return "\n".join(algo_output.terminal_state.action_history)
    except Exception as e:
        print("Error in output extraction,", e)
        return ""


BWAction = str
class BWState(NamedTuple):
    """The state of the Blocksworld for ToT

    See the docstring of BlocksWorldModel for more details.
    """
    step_idx: int
    action_history: list[str]
    end: bool

    def __hash__(self):
        return hash((self.step_idx, tuple(self.action_history), self.end))

    # Implementing __eq__ for object comparison
    def __eq__(self, other):
        if not isinstance(other, BWState):
            return False
        return (self.step_idx == other.step_idx and
                tuple(self.action_history) == tuple(other.action_history) and
                self.end == other.end)

class BWConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 temperature: float = 0.8,
                 n_candidate: int = 4) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.n_candidate = n_candidate
        self.temperature = temperature

    def get_prompt_by_state(self, state: BWState) -> str:
        return self.prompt["icl"].replace("<action>", "\n".join(state.action_history + [""])) \
            .replace("<init_state>", utils.extract_init_state(self.example)) \
            .replace("<goals>", utils.extract_goals(self.example, return_raw=True))


    def get_actions(self, state: BWState) -> list[BWAction]:
        return self.get_actions_generic(state)

    def get_actions_generic(self, state, use_beam_search_outputs=False, do_sample=True):
        """Generalization to get_actions method, allows to set various parameters to thought generation,
        like beam search that create exactly top n thoughts deterministically, and maybe more options in the future."""
        prompts = self.get_prompt_by_state(state)
        num_beams = self.n_candidate if use_beam_search_outputs else None
        outputs = self.base_model.generate([prompts],
                                           num_return_sequences=self.n_candidate,
                                           # max_length=20,
                                           eos_token_id=["\n", ],
                                           temperature=self.temperature,
                                           do_sample=do_sample,
                                           hide_input=True,
                                           use_beam_search_outputs=use_beam_search_outputs,
                                           num_beams=num_beams).text

        outputs = [output.split("\n")[0] for output in outputs]
        # deduplicate
        outputs = list(dict.fromkeys(outputs))
        if '' in outputs:
            outputs.remove('')
        return outputs


    def get_pi(self, state:BWState, actions: list[BWAction], temperature=None):
        """
        TODO: log prob to prob conversion
        """
        temperature = self.temperature if temperature is None else temperature

        inputs = self.prompt["icl"].replace("<action>", "\n".join(state.action_history + [""])) \
            .replace("<init_state>", utils.extract_init_state(self.example)) \
            .replace("<goals>", utils.extract_goals(self.example, return_raw=True))
        
        log_probs = self.base_model.get_loglikelihood(inputs, 
                        [inputs + action for action in actions], temperature=temperature)
        
        probs = np.exp(log_probs)
        print(probs)
        return probs

    def fast_reward(self, state: BWState, action: BWAction) -> tuple[float, dict]:
        inputs = self.prompt["icl"].replace("<action>", "\n".join(state.action_history + [""])) \
            .replace("<init_state>", utils.extract_init_state(self.example)) \
            .replace("<goals>", utils.extract_goals(self.example, return_raw=True))[:-1]
        
        intuition = self.base_model.get_loglikelihood(inputs+ "\n", [inputs + "\n" + action])[0]

        self_eval_prompt = self.prompt["self-eval"].replace("<init_state>", 
                                                            utils.extract_init_state(self.example)) \
                                                   .replace("<goals>", utils.extract_goals(self.example, return_raw=True)) \
                                                   .replace("<action>", action)
        self_eval = self.base_model.get_loglikelihood(self_eval_prompt, 
            [self_eval_prompt + "good"])[0]

        return intuition + self_eval, {'intuition': intuition, "self_eval": self_eval}

    def reward(self, state: BWState, action: BWAction, **kwargs) -> tuple[float, dict]:
        # since these two rewards are fast, we can just return the reward
        intuition, self_eval = kwargs['intuition'], kwargs['self_eval']
        return intuition + self_eval, {'intuition': intuition, "self_eval": self_eval}

    def update_example(self, example, prompt=None) -> None:
        super().update_example(example, prompt=prompt)

class BlocksWorldModel(WorldModel):
    """Blocks World World Model
    State: (step_idx, action_history: [str])
    Action: e.g. "put the red block on the green block"
    """

    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 max_steps: int = 6,
                 batch_size=8) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.base_model = base_model
        self.prompt = prompt  # need to check if this is necessary
        self.batch_size = batch_size

    def init_state(self) -> BWState:
        """Initialize the world model.

        :return: the initial state
        """
        return BWState(step_idx=0, action_history=[], end=False)

    def step(self, state: BWState, action: BWAction) -> tuple[BWState, dict]:
        """Take a step in the world model.
        
        :param state: the current state (see the docstring of BlocksWorldModel)
        :param action: the action to take
        :return: the next state and additional information cached for reward calculation
        """
        state = copy.deepcopy(state)
        if action != "[PLAN END]" and action != '':
            state = BWState(step_idx=state.step_idx + 1, action_history=state.action_history + [action], end=False)
        else:
            state = BWState(step_idx=state.step_idx + 1, action_history=state.action_history, end=True)
        return state, {}

    def is_terminal(self, state: BWState) -> bool:
        if state.end:
            return True
        elif state.step_idx == self.max_steps:
            return True
        return False

def tot_bw(base_model: LanguageModel,
           prompt: dict,
           search_algo: str = "beam",
           data_path: str = 'data',
           resume: int = 0,
           depth_limit: int = 6,
           log_dir: Optional[str] = None,
           disable_log: bool = False,
           domain_file: str = "",
           config_file: str = "",
           lm_plan_file: str = 'lm_plan.tmp',
           temperature: float = 0.8,
           **search_algo_params):

    if search_algo == "beam":
        search_algo_params |= {"max_depth": depth_limit}
    elif search_algo == "dfs":
        search_algo_params |= {"depth": depth_limit}
    elif search_algo == "lts":
        pass
        #search_algo_params |= {"depth": depth_limit}
    else:
        print("Unknown search algorithm", search_algo)
        raise NotImplementedError
    world_model = BlocksWorldModel(base_model=base_model, prompt=prompt, max_steps=depth_limit)
    config = BWConfig(base_model=base_model, prompt=prompt, temperature=temperature)
    
    if search_algo == "dfs":
        output_extractor = dfs_bw_extractor
        search_algo = DFS(**search_algo_params)
    elif search_algo == "beam":
        output_extractor = bfs_bw_extractor
        search_algo = BeamSearch(**search_algo_params)
    elif search_algo == "lts":
        output_extractor = lts_bw_extractor
        search_algo = LTS(**search_algo_params)
    else:
        raise NotImplementedError
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
    evaluator = BWEvaluator(config_file=config_file, domain_file=domain_file, data_path=data_path, init_prompt=prompt, disable_log=disable_log, output_extractor=output_extractor)
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(accuracy)

if __name__ == '__main__':
    import os
    import sys
    import json
    import warnings
    import fire
    import random
    import torch
    import torch.backends.cudnn
    from reasoners.lm import LlamaModel, Llama2Model
    from reasoners.lm import OpenAIModel
    from reasoners.lm.llama_model import DummyLLaMAModel
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    def main(
            base_lm: Literal[ 'llama2',' exllama', 'llama3', 'openai']  = 'openai',
            model_dir = '/path/to/model',
            llama_size = "7B",
            lora_dir = None,
            prompt_path: str = 'examples/CoT/blocksworld/prompts/pool_prompt_v1.json',
            data_path: str = 'examples/CoT/blocksworld/data/split_v1/split_v1_step_2_data.json',
            disable_log: bool = False,
            config_file: str = "examples/CoT/blocksworld/data/bw_config.yaml",
            domain_file: str = "examples/CoT/blocksworld/data/generated_domain.pddl",
            lm_plan_file: str = 'lm_plan.tmp',
            depth_limit: int = 6,
            mem_map = None,
            temperature = 0.8,
            search_algo = "beam",
            batch_size = 4,
            **kwargs
            ):
        print(model_dir)
        with open(prompt_path) as f:
            prompt = json.load(f)
        
        print(batch_size)
        if base_lm in ['llama2', 'llama3']:    
            # import torch
            # import torch.backends.cudnn
            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True

        if base_lm == 'llama2':
            from reasoners.lm import Llama2Model
            model = Llama2Model(model_dir, llama_size, max_batch_size=batch_size)
        elif base_lm == 'llama3' or base_lm == 'llama3.2':
            from reasoners.lm import Llama3Model
            model = Llama3Model(model_dir, llama_size, max_batch_size=batch_size)
        elif base_lm == 'openai':
            ######## TODO: Code Here  ###########
            #model = OpenAIModel(model='gpt-4o-mini', temperature=temperature, max_tokens=2048 )
            model = OpenAIModel(model='gpt-3.5-turbo', temperature=temperature, max_tokens=2048 )
            #####################################
        elif base_lm == 'hf':
            from reasoners.lm import HFModel
            model = HFModel(model_dir, model_dir, max_batch_size=batch_size)
        else:
            from reasoners.lm import ExLlamaModel  # Maybe other transformer models also support
            device = torch.device("cuda:0")
            model = ExLlamaModel(model_dir, 
                                    lora_dir, 
                                    device=device, 
                                    max_batch_size=batch_size, 
                                    max_new_tokens=200, 
                                    max_seq_length=2048, 
                                    mem_map=mem_map,
                                    log_output=True)#please set mem_map if you need model parallelism, e.g. mem_map = [16,22] with 2 GPUs

        tot_bw(model,
               prompt,
               search_algo=search_algo,
               disable_log=disable_log,
               data_path=data_path,
               config_file=config_file,
               domain_file=domain_file,
               depth_limit=depth_limit,
               lm_plan_file=lm_plan_file,
               temperature=temperature, **kwargs)
    
    fire.Fire(main) # user will need to switch the model in the code


'''



CUDA_VISIBLE_DEVICES=2,3 python examples/tot/blocksworld/tot_inference.py --data_path 'examples/tot/blocksworld/data/split_v1/split_v1_step_2_data.json' --mem_map "[16,22]" --depth_limit 2 --model_dir $LLAMA2_CKPTS --prompt_path examples/tot/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/bfs_v1_step2_f --beam_size 10 --temperature 0.8 --reward_aggregator mean | tee debug_bfs.log

CUDA_VISIBLE_DEVICES=0,1 python examples/tot/blocksworld/tot_inference.py --data_path 'examples/tot/blocksworld/data/split_v1/split_v1_step_4_data.json' --mem_map "[16,22]" --depth_limit 4 --model_dir $LLAMA2_CKPTS --prompt_path examples/tot/blocksworld/prompts/pool_prompt_v1.json --log_dir logs/dfs_v1_step4_ff --temperature 0.8 --search_algo dfs --total_states 10 --max_per_state 3 | tee debug_dfs.log
'''
