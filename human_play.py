# -*- coding: gb2312 -*-
from __future__ import print_function
import pickle
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
import torch
from train import Net
from policy_value_net_pytorch import PolicyValueNet  



class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)



if __name__ == '__main__':
    n=4
    width,height=6,6
    model_file='best_policy.model'

    #create UI
    board = Board(width=width,height=height,n_in_row=n)
    #create game agent
    game = Game(board)
    
    #load trained model
    #best_policy = PolicyValueNet(width, height, model_file = model_file)
    policy_value_net = Net(6, 6).cuda()
    policy_value_net.load_state_dict(torch.load(model_file, encoding='latin1'))
    mcts_player = MCTSPlayer(policy_value_net, c_puct=5, n_playout=400)

    # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
    # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

    human=Human()
    game.start_play(human,mcts_player,start_player=1,is_shown=1)

