# -*- coding: utf-8 -*-
import numpy as np
import copy
from operator import itemgetter

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
        return max(self._children.items(),key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits+=1
        #Q=((n-1)*Q+value)/n
        self._Q+=(leaf_value-self._Q)/self._n_visits

    def update_recursive(self, leaf_value):
        #back to root
        if self._parent: self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u=(c_puct*self._P*np.sqrt(self._parent._n_visits)/(1+self._n_visits))
        return self._Q+self._u

    def is_leaf(self):
        return self._children=={}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    def __init__(self, c_puct=5, n_playout=10000):
        #c_puct:  A higher value means relying on the prior more.
        self._root=TreeNode(None,1.0)
        self._c_puct=c_puct
        self._n_playout=n_playout

    def _playout(self, state):
        node=self._root
        while(1):
            #walk from root to leave
            if node.is_leaf():  break
            #choice biggest value action
            action,node=node.select(self._c_puct)
            state.do_move(action)

        action_probs=zip(state.availables,np.ones(len(state.availables))/len(state.availables))
        
        #we just walk into the leave of current tree,if the current tree isn't deep,it don't means end.
        end,_=state.game_end()
        #walk start from leave,just walk one step
        if not end: node.expand(action_probs)
        #random walk deeo from leave obeserve whether win or not
        leaf_value=self._evaluate_rollout(state)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self,state,limit=1000):
        #we don't record,just want to now whether could win.
        player=state.get_current_player()
        for _ in range(limit):
            end,winner=state.game_end()
            if end: break
            action_probs=zip(state.availables,np.random.rand(len(state.availables)))
            max_action=max(action_probs,key=itemgetter(1))[0]
            #we should remove the this position in avaliable,since we can use it twise.
            state.do_move(max_action)

        if winner==-1: return 0
        else: return 1 if winner==player else -1

    def get_move(self,state):
        #keep the origin state and do many game
        for _ in range(self._n_playout):
            #since playout will change the state,we should deepcopy first
            state_copy=copy.deepcopy(state)
            self._playout(state_copy)
        #do which action we do most before
        return max(self._root._children.items(),key=lambda act_node:act_node[1]._n_visits)[0]


class MCTSPlayer(object):
    def __init__(self,c_puct=5,n_playout=2000):
        self.mcts=MCTS(c_puct,n_playout)

    def set_player_ind(self, p):
        self.player=p

    def reset_player(self):
        self.mcts._root=TreeNode(None,1.0)

    def get_action(self,board):
        sensible_moves=board.availables
        #the board isn't full.
        if len(sensible_moves)>0:
            move=self.mcts.get_move(board)
            self.mcts._root=TreeNode(None,1.0)
            return move
        else:
            print("WARNING: the board is full")

