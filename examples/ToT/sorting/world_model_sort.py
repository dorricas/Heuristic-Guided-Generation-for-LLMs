import sys
from dataclasses import dataclass
from typing import Optional

import copy
import prompts.output
import prompts.transition
from reasoners import WorldModel, LanguageModel
from reasoners.base import Example
#from examples.prontoqa.dataset import ProntoQAExample

# TODO 
from sorting.prompts.dataset import SortExample

'''
@dataclass
class SortState:
    
    input: str
    current: str
    output: Optional[str] = None

    def __str__(self):
        if self.output is None:
            return f'SortState({repr(self.current)})'
        else:
            return f'SortState({repr(self.current)}, output={repr(self.output)})'

    def __hash__(self):
        return hash(self.current)
'''

SortState = str
SortAction = str




class SortWorldModel(WorldModel[SortState, SortAction]):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: str,
                 temperature) -> None:
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt
        self.temperature = temperature

    def init_state(self) -> SortState:

        return SortState(self.example)

    def step(self, state: SortState, action: SortAction) -> tuple[SortState, dict]:

        # call the LLM to predict the state transition
        state = copy.deepcopy(state)
        current_prompt = self.prompt + state
        new_state = self.base_model.generate([current_prompt],
                                eos_token_id="\n", hide_input=True, temperature=self.temperature).text[0].strip()
        
        print(new_state)
        return SortState(body=new_state, last_state=state, last_action=action), {}

    def is_terminal(self, state: SortState) -> bool:

        return "Answer:" in state
