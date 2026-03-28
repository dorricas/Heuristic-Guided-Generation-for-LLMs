import heapq
from .. import SearchAlgorithm, WorldModel, Reasoner, SearchConfig, State, Action
from typing import List, Optional, Tuple, NamedTuple, Generic
import itertools
EPSILON=1e-14

class LTSNode:
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self, state: Optional[State], action: Optional[Action],
                 parent: "Optional[LTSNode]" = None, pi: float = 0.0,
                 is_terminal: bool = False) -> None:
        "Track prob or log prob?"
        self.id = next(LTSNode.id_iter)
        self.state = state
        self.action = action
        self.parent = parent
        self.children: 'Optional[list[LTSNode]]' = []
        self.cum_rewards = []
     
        self.g = 0
        self.pi = 1
        if parent:
            self.g = self.parent.g + 1     #cost from start to this node
            self.pi = self.parent.pi
        self.pi *= pi  # heuristic estimate
        self.f = self.g / (self.pi + EPSILON) #for stability
        self.is_terminal = is_terminal

    def add_child(self, child: 'LTSNode'):
        self.children.append(child)
    
    def __lt__(self, other):
        return self.f < other.f

    def get_trace(self) -> List[Tuple[Action, State, float]]:
        """ Returns the sequence of actions and states from the root to the current node """
        node, path = self, []
        while node is not None:
            path.append((node.action, node.state, node.g))  # g or self.f?
            node = node.parent
        return path[::-1]  # Reverse to get the correct order

class LTSResult(NamedTuple):
    terminal_state: State
    cum_rewards: float
    tree_state: LTSNode
    terminal_nodes: List[LTSNode]

class LTS(SearchAlgorithm, Generic[State, Action]):
    """ LTS Search Algorithm """

    def __init__(self, total_calls: int=100, max_per_state: int=10, 
                 lts_temp=0.8, max_terminal_nodes: int=10):
        self.total_calls = total_calls
        self.terminals = []
        self.call_cnt = 0
        self.max_per_state = max_per_state
        self.max_terminal_nodes = max_terminal_nodes ## TODO: Redundant as of now
        self.anytime = False 
        self.lts_temp = lts_temp

    def _reset(self):
        self.terminals = []
        self.call_cnt = 0

    def __call__(self, world: WorldModel, config: SearchConfig) -> Optional[LTSResult]:
        # Reset the node IDs
        LTSNode.reset_id()

        # Initialize the search
        init_state = world.init_state()
        self._reset()
        
        # Create the root node with the initial state
        init_node = LTSNode(
            state=init_state, 
            action=None, 
            parent=None, 
            pi=1,
            is_terminal=False
        )
        
        # Priority queue (min-heap) to select nodes based on their f
        open_set = []
        heapq.heappush(open_set, init_node)
        
        # Set to track visited states
        visited = set()

        while open_set and self.call_cnt < self.total_calls:
            # Get the node with the lowest f_cost
            cur_node = heapq.heappop(open_set)

            # If the current node represents a terminal state, return the result
            if world.is_terminal(cur_node.state):
                self.terminals.append(cur_node) #TODO: Wait until max_terminals?
            
            if len(self.terminals) == 1 and not self.anytime:
                break
           
            # -- removed -- no condition on terminal nodes for now
            # if len(self.terminals) == self.max_terminal_nodes and self.anytime:
            #    break

            # Mark the current state as visited
            visited.add(cur_node.state)

            # Get possible actions from the current state
            new_actions = config.get_actions(cur_node.state)
            if not new_actions:
                continue  # No actions to explore, skip this node

            if len(new_actions) > self.max_per_state:
                new_actions = new_actions[:self.max_per_state]

            max_num_new_actions = min(len(new_actions), self.total_calls - self.call_cnt)
            new_actions = new_actions[:max_num_new_actions]
            self.call_cnt += len(new_actions)
            print(new_actions)

            if not new_actions:
                continue  # No actions to explore, skip this node

            if len(new_actions) > self.max_per_state:
                new_actions = new_actions[:self.max_per_state]

            pis = config.get_pi(cur_node.state, new_actions, self.lts_temp)  #Save computational cost
             # Explore each action
            for itr, action in enumerate(new_actions):
                # Generate the next state
                #action, (fast_reward, fast_reward_details) = action
                new_state, aux = world.step(cur_node.state, action)
                
                # Skip already visited states
                if new_state in visited:
                    continue

                # Compute costs
                pi = pis[itr]

                # Create a new node for the new state
                new_node = LTSNode(
                    state=new_state, 
                    action=action, 
                    parent=cur_node, 
                    pi=pi, 
                    is_terminal=world.is_terminal(new_state)
                )
                fast_reward, fast_reward_details = config.fast_reward(cur_node.state, action)
                reward, reward_details = config.reward(cur_node.state, action, **fast_reward_details) 
                new_node.cum_rewards = cur_node.cum_rewards + [reward]


                # Add the new node to the priority queue
                heapq.heappush(open_set, new_node)

                cur_node.add_child(new_node)

        #DEBUG
        print("LTS found goal nodes in", self.call_cnt, "LLM calls")
        print("Num Terminals:", len(self.terminals))
        # If terminal states are found, return the best one based on reward 
        if self.terminals:
            best_terminal = max(self.terminals, key=lambda x: sum(x.cum_rewards))
            result = LTSResult(
                terminal_state=best_terminal.state,
                cum_rewards=sum(best_terminal.cum_rewards),
                tree_state=init_node,
                terminal_nodes=self.terminals) 
            return result
        return None  # No terminal state found

