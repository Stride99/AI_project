import numpy as np
import copy
import torch
from torch.autograd import Variable

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent=parent
        self._children={} 
        self._n_visits=0
        self._Q=0
        self._u=0
        self._P=prior_p

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action]=TreeNode(self, prob)

    def select(self, c_puct):
        return max(self._children.items(),key=lambda act_node: act_node[1]._Q+(c_puct*act_node[1]._P*np.sqrt(act_node[1]._parent._n_visits)/(1+act_node[1]._n_visits)))

    def update(self, leaf_value):
        #back to root
        if self._parent: self._parent.update(-leaf_value)
        self._n_visits+=1
        #Q=((n-1)*Q+value)/n
        self._Q+=(leaf_value-self._Q)/self._n_visits


class MCTS(object):
    def __init__(self,policy_value_net,n_playout=10000):
        self._root=TreeNode(None,1.0)
        self.policy_value_net=policy_value_net
        self._n_playout=n_playout

    def _playout(self, state):
        node=self._root
        while(1):
            if node._children=={}:  break
            #choice biggest value action
            action,node=node.select(5)
            state.do_move(action)

        board=state
        legal_positions=board.availables
        current_state=np.ascontiguousarray(board.current_state().reshape(-1,4,6,6))
        log_act_probs,value=self.policy_value_net(Variable(torch.from_numpy(current_state)).cuda().float())
        #since before we record log,so we should return it back
        act_probs=np.exp(log_act_probs.data.cpu().numpy().flatten())
        act_probs=zip(legal_positions,act_probs[legal_positions])
        value=value.data[0][0]
        action_probs,leaf_value=act_probs,value
    
        player=state.get_current_player()
        end, winner=state.game_end()
        if not end: node.expand(action_probs)
        else:
            if winner==-1: leaf_value=0
            else: leaf_value=1 if winner==player else -1

        node.update(-leaf_value)

    def get_move_probs(self,state):
        #Play n games
        for _ in range(self._n_playout):
            state_copy=copy.deepcopy(state)
            self._playout(state_copy)

        act_visits=[(act,node._n_visits)for act,node in self._root._children.items()]
        acts,visits=zip(*act_visits)
        act_probs=softmax(np.log(np.array(visits)+1e-10)*1000)

        return acts, act_probs


class MCTSPlayer(object):

    def __init__(self,policy_value_net,n_playout=2000,is_selfplay=0):
        self.mcts=MCTS(policy_value_net,n_playout)
        self._is_selfplay=is_selfplay


    def get_action(self,board,return_prob=0):
        move_probs=np.zeros(board.width*board.height)
        if len(board.availables)>0:
            acts,probs=self.mcts.get_move_probs(board) 
            move_probs[list(acts)]=probs
            if self._is_selfplay:
                #use dirichlet noise to exploration 
                move=np.random.choice(acts,p=0.75*probs+0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
                # update the root node and reuse the search tree
                self.mcts._root=self.mcts._root._children[move]
                self.mcts._root._parent=None
            else:
                move=np.random.choice(acts,p=probs)
                self.mcts._root=TreeNode(None,1)
            if return_prob:
                return move,move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")