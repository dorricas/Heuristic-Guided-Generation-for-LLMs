from typing import NamedTuple

import torch

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners import WorldModel, LanguageModel, SearchConfig


class TLSState(NamedTuple):
    """State of token level search. represented as a sequence of tokens"""
    step_idx: int
    action_history: list[int]
    end: bool

    def __hash__(self):
        return hash((self.step_idx, tuple(self.action_history), self.end))

    # Implementing __eq__ for object comparison
    def __eq__(self, other):
        if not isinstance(other, TLSState):
            return False
        return (self.step_idx == other.step_idx and
                tuple(self.action_history) == tuple(other.action_history) and
                self.end == other.end)

    def get_text(self, model: LanguageModel) -> str:
        return model.tokenizer.decode(torch.tensor(self.action_history))



class TLSConfig(SearchConfig):
    """Token level search configuration"""
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 # temperature: float,
                 k: int) -> None:

        super().__init__()
        self.base_model = base_model
        # self.example = None
        # self.temperature = temperature
        self.prompt = prompt
        self.k = k

    def get_actions(self, state: TLSState) -> list[int]:
        tokens = torch.tensor([state.action_history], device=self.base_model.device)
        with torch.no_grad():
            logits = self.base_model.model.forward(
                tokens,
                self.base_model.cache,
                last_id_only=True,
                preprocess_only=False,
                lora=self.base_model.lora,
                output_device=self.base_model.device,
                # input_mask=torch.ones_like(tokens)
            )

        _, topk_indices = torch.topk(logits[0].squeeze(1), self.k)
        return topk_indices[0].tolist()

    def get_action_tokens(self, state: TLSState) -> list[str]:
        action_indices = self.get_actions(state)
        action_indices_tensor = torch.tensor(action_indices, device=self.base_model.device).unsqueeze(1)
        return self.base_model.tokenizer.decode(action_indices_tensor)


    def reward(self, state: TLSState, action: str, **kwargs) -> tuple[float, dict]:
        pass


class TLSWorldModel(WorldModel):
    """Token level search World Model"""
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 max_steps: int = 6,
                 batch_size=8) -> None:

        super().__init__()
        self.max_steps = max_steps
        self.base_model = base_model
        self.prompt = prompt

    def init_state(self) -> TLSState:
        """Initialize the world model.

        :return: the initial state
        """
        return TLSState(step_idx=0, action_history=[], end=False)

    def step(self, state: TLSState, action: str) -> tuple[TLSState, dict]:
        """Take a step in the world model.

        :param state: the current state
        :param action: the action to take (the next token to choose)
        :return: the next state and additional information cached for reward calculation
        """
        state = copy.deepcopy(state)
        if action != "[PLAN END]" and action != '':
            state = TLSState(step_idx=state.step_idx + 1, action_history=state.action_history + [action], end=False)
        else:
            state = TLSState(step_idx=state.step_idx + 1, action_history=state.action_history, end=True)
        return state, {}

    def is_terminal(self, state: TLSState) -> bool:
        if state.end:
            return True
        elif state.step_idx == self.max_steps:
            return True
        return False
