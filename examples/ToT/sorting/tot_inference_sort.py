from reasoners.lm import ExLlamaModel
import json
import fire
from typing import Sequence, Any
import json
from tqdm import tqdm
from typing import Type, Callable, Optional, Literal

#from dataset import ProntoQADataset, ProntoQAExample
from reasoners import Reasoner
import torch

# TODO
#import prompts.finish
#import prompts.next_step
#import prompts.valid_tot
#from dataset import SortDataset, SortExample
from examples.ToT.sorting.prompts.prompt import sort_example

import numpy as np
import random
from reasoners import WorldModel, SearchConfig
from reasoners.algorithm import BeamSearch, DFS, LTS, MCTS
#from reasoners.benchmark import ProntoQAEvaluatorFinal

# TODO 
#from reasoners.benchmark import SortEvaluatorFinal

SortAction = str

class SortState:
    def __init__(self, states: list[str]):
        self.states = tuple(states)  # Convert to immutable tuple

    def __eq__(self, other):
        if isinstance(other, SortState):
            return self.states == other.states
        return False

    def __hash__(self):
        return hash(self.states)  # Hash based on the tuple

    def __len__(self):
        return len(self.states)

    def __iter__(self):
        return iter(self.states)  # Delegate iteration to the list

    def __getitem__(self, index):
        return self.states[index]  # Delegate indexing to the underlying list
    
    def __str__(self):
        return str(self.states)

    def take_action(self, action: SortAction):
        return SortState(list(self.states) + [action])

def remove_so_prefix(s):
    if s.startswith('So '):
        return s[3:]
    return s


from dataclasses import dataclass, field
from typing import List, Dict, Iterator, Union, Mapping

@dataclass
class SortProblem:
    question: str
    query: str
    chain_of_thought: List[str]
    answer: str


@dataclass
class SortExample:
    # in_context_examples: Mapping[str, SortProblem]
    # test_example: SortProblem = field(default=None)
    example = sort_example





class SortToTWorldModel(WorldModel[SortState, SortAction, SortExample]):
    def __init__(self) -> None:
        super().__init__()
    
    def init_state(self) -> SortState:
        return SortState([])
    
    def step(self, state: SortState, action: SortAction) -> tuple[SortState, dict]:
        return state.take_action(action), {}
    
    def is_terminal(self, state: SortState) -> bool:
        if len(state) > 0 and "Answer:" in state[-1]:
            return True
        return False
    

from reasoners import Evaluator



def get_cot_prompt(sampled_data):
    # formatted_examples = ""
    # for i, entry in enumerate(sampled_data, 1):
    #     formatted_examples += f"Q: {entry['Facts']} {entry['claims'][0]} {entry['Query']}\n"
    #     formatted_examples += f"A: {entry['claims'][0]} "
    #     for j, (claim, next_step) in enumerate(zip(entry['claims'][1:], entry['next_steps'][:-1]), 1):
    #         formatted_examples += f"{next_step} So {claim} "
    #     tf = not (("not" in entry['claims'][-1]) ^ ("not" in entry['Query']))
    #     formatted_examples += f"The answer is {'true' if tf else 'false'}.\n\n"
    # return formatted_examples
    return



class SortEvaluatorFinal(Evaluator):
    def __init__(self, 
                 output_extractor = lambda x: x.terminal_state.body if x.terminal_state is not None else "",
                 answer_extractor = lambda x: x.test_example.answer,
                 init_prompt=None,
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="cot", 
                 dataset=None) -> None:

        dataset_list = list(dataset)
        dataset_list = dataset_list
        self.queries = [obj[0] for obj in dataset_list]
        self.dataset = iter(dataset_list)
        self.answers = [obj[1] for obj in dataset_list]
        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.input_processor = lambda x: x
        self.full_dataset = list(dataset_list)
        self._dataset_name = 'sorting'
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm
        self.sample_prompt_type = sample_prompt_type

    def sample_prompt(self,
                      shuffle_prompt=True,
                      num_shot=4):
        if shuffle_prompt:
            ret = random.sample(list(self.init_prompt), k=num_shot)
        else:
            ret = self.init_prompt[:num_shot]

        if self.sample_prompt_type == "rap":
            return ret

        elif self.sample_prompt_type == "cot":
            # cot or tot
            return get_cot_prompt(ret)
        else:
            raise NotImplementedError

    def eval_output(self, answer, output):
        
        def compare(output, answer):
            # Extract content between brackets, handling various formats
            out_str = output.split("[")[1].split("]")[0]
            ans_str = answer.strip().strip("[]")
            
            # Parse to floats, .strip() handles extra spaces around numbers
            out = [float(x.strip()) for x in out_str.split(",") if x.strip()]
            ans = [float(x.strip()) for x in ans_str.split(",") if x.strip()]
            print("Eval:", out, ans) 
            return out == ans
        
        if output is None:
            return False
        try:
            output = str(output)
            answer = str(answer)
            return compare(output, answer)
        except ValueError:
            return output == answer
    

class SortToTSearchConfig(SearchConfig[SortState, SortAction, SortExample]):
    def __init__(self, base_model, prompt, n_actions=5, temperature=0.8) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.temperature = temperature
        self.base_model = base_model
        self.prompt = prompt
        assert temperature > 0, "Temperature = 0 indicates greedy decoding. There is no point running multiple chains"

    def get_actions(self, state: SortState) -> list[SortAction]:
        # print(f"state: {state}\n")
        input_prompt = self.prompt
        

        input_prompt += "Input: " + self.example[0]+ "\nSteps:\n"
        
        #print(f"Input prompt: '{input_prompt}'\n")
        #print("#"*100)
        
        input_prompt += "".join([" " + s for s in state])
        input_prompt += "\n"
        #eos_token_id=29889
        eos_token_id=["\n"]
        output = self.base_model.generate([input_prompt] * self.n_actions, eos_token_id=eos_token_id, hide_input=True, temperature=self.temperature, do_sample=True).text
        ret = [o.split("\n")[0].strip() for o in output]
        #print(f"Input prompt to model.generate: {input_prompt}")
        print(f"Model generated actions: {ret}")
        # deduplicate
        ret = list(dict.fromkeys(ret).keys())
        
        if '' in ret:
            ret.remove('')

        return ret

    def fast_reward(self, state: SortState, action: SortAction) -> tuple[float, dict]:
        processed_state = [remove_so_prefix(s) for s in state]
        processed_action = remove_so_prefix(action)
        input_prompt = self.prompt
        input_prompt += "Q: " + self.example[0] + "\nSteps:\n"
        input_prompt += "".join([" " + s for s in processed_state])
        candidate = input_prompt + " " + processed_action
        intuition = self.base_model.get_loglikelihood(input_prompt, 
            [candidate])[0]
        
        return  (intuition.item(), {'intuition':intuition})  
        
        #print(f" prompt: {self.prompt}")
        #print(f"action: {processed_action}")
        #print(f"input_prompt: {input_prompt}")
        #print("hello")
        #print(f"state: {processed_state}")

        # input_prompt = ""
        # input_prompt += prompts.valid_tot.EXAMPLES
        # input_prompt += prompts.valid_tot.FACTS_FORMAT.format(self.example.test_example.question or "", self.example.test_example.query)
        # input_prompt += prompts.valid_tot.NEXT_STEP_FORMAT.format(',\n'.join(f'"{statement}"' for statement in processed_state))
        # input_prompt += prompts.valid_tot.VALID_PREFIX

        # output_logits = self.base_model.get_next_token_logits(
        #     input_prompt,
        #     candidates=["Yes", "No"]
        # )

        # #print(f"input_prompt: {input_prompt}")
        # reward: float = output_logits[0][0].item()
        # reward:float = torch.softmax(torch.tensor(output_logits[0]), dim=0)[0].item()
        # #print(f" reward: {reward}")

        # self_eval = reward  
        # #print(f" intuition: {intuition}, self_eval: {self_eval}")
        # return intuition*0.5 + self_eval*0.5, {"intuition": intuition, "self_eval":self_eval}

    def reward(self, state, action, **kwargs) -> tuple[float, dict]:
        # how correct is this last action
        intuition = kwargs["intuition"]
        #self_eval = kwargs["self_eval"]
        #return intuition*0.5 + self_eval*0.5, {"intuition": intuition, "self_eval":self_eval}
        return intuition*0.5, {"intuition": intuition}
    
    def get_pi(self, state, actions, temperature=None):
        """
        TODO: log prob to prob conversion
        """
        temperature = self.temperature if temperature is None else temperature
        input_prompt = self.prompt
        input_prompt += "Input: " + self.example[0]+ "\nSteps:\n"
        
        #print(f"Input prompt: '{input_prompt}'\n")
        #print("#"*100)
        
        input_prompt += "".join([" " + s for s in state])
        input_prompt += "\n"

        log_probs = self.base_model.get_loglikelihood(input_prompt, 
                        [input_prompt + ' ' + action for action in actions], temperature=temperature)
        
        probs = np.exp(log_probs)
        print(actions)
        print(probs)
        return probs


def main(
           model_dir: str,
           base_lm: Literal[ 'llama2',' exllama', 'llama3', 'openai']  = 'exllama',
           llama_size = "7B",
           batch_size = 5,
           search_algo: str = "beam",
           resume: int = 0,
           depth_limit: int = 6,
           log_dir: Optional[str] = None,
           temperature: float = 0.8,
           mem_map: str = [16, 22],
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
    

    def extractor(algo_output):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        # to make sure the plan is saved before evaluation in multi-process setting
        try:
            #print("Extractor", algo_output.terminal_nodes[0].state)
            #print("Extractor", algo_output.terminal_nodes[0].state[-1])
            answer = algo_output.terminal_nodes[0].state[-1]
            #answer = answer.replace("Answer: ", "")
            return answer

        except Exception as e:
            print("Error in output extraction,", e)
            return ""
    
    if base_lm in ['llama2', 'llama3']:    
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True

    if base_lm == 'llama2':
        from reasoners.lm import Llama2Model
        model = Llama2Model(model_dir, llama_size, max_batch_size=batch_size)
    elif base_lm == 'llama3' or base_lm == "llama3.2":
        from reasoners.lm import Llama3Model
        model = Llama3Model(model_dir, llama_size, max_batch_size=batch_size)
    elif base_lm == 'openai':
        from reasoners.lm import OpenAIModel
        model = OpenAIModel(model='gpt-3.5-turbo', temperature=temperature, max_tokens=2048 )
    elif base_lm == 'hf':
        from reasoners.lm import HFModel
        model = HFModel(model_dir, model_dir, max_batch_size=batch_size)
    else:
        from reasoners.lm import ExLlamaModel  # Maybe other transformer models also support
        model = ExLlamaModel(model_dir, 
                                lora_dir=None, 
                                device=torch.device("mps"), 
                                max_batch_size=1, 
                                max_new_tokens=200, 
                                max_seq_length=2048, 
                                mem_map=mem_map)

    world_model = SortToTWorldModel()
    search_config = SortToTSearchConfig(base_model=model, prompt = sort_example, temperature=temperature)
    
    output_extractor = extractor
    if search_algo == "dfs":
        search_algo = DFS(**search_algo_params)
    elif search_algo == "beam":
        search_algo = BeamSearch(**search_algo_params)
    elif search_algo == "lts":
        search_algo = LTS(**search_algo_params)
    else:
        raise NotImplementedError
   
    # with open('examples/CoT/sort/data/example_next_steps.json') as f:
    #     init_prompt = json.load(f)

    init_prompt = sort_example
    with open("examples/ToT/sorting/dataset/dataset.json", "r") as f:
        data = json.load(f)
    dataset = [[entry["input"], entry["answer"]] for entry in data.values()]
    
    reasoner = Reasoner(world_model=world_model, search_config=search_config, search_algo=search_algo)
    evaluator = SortEvaluatorFinal(
        init_prompt=init_prompt,
        sample_prompt_type="cot",
        disable_log=False,
        disable_tqdm=False, 
        dataset = dataset,
        output_extractor=output_extractor,
        #answer_extractor=lambda x: "\n".join(x.test_example.chain_of_thought[2::2])
        answer_extractor=lambda x: x[1] 
    )

    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
    print(accuracy)

if __name__ == '__main__':
    fire.Fire(main)

# CUDA_VISIBLE_DEVICES=0 python examples/tot/prontoqa/inference_tot.py --depth_limit 10 --model_dir $LLAMA2_CKPTS --beam_size 10 --temperature 0.8 --reward_aggregator mean --search_algo beam > debug_bfs.log

# python examples/rap_prontoqa/tot_inference.py --depth_limit 10 --model_dir /data/yi/Llama-2-70B-GPTQ/ --total_states 10 --temperature 0.8 --search_algo dfs --max_per_state 3 > debug_dfs.log
    
    # TODO: 1) remove total state, depth limit 2) 
# python examples/tot/prontoqa/tot_inference.py --depth_limit 10 --model_dir /data/yi/Llama-2-70B-GPTQ/ --total_states 10 --temperature 0.8 --search_algo dfs --max_per_state 3 > debug_dfs.log
