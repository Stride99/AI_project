# -*- coding: utf-8 -*-
from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height

        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height,board_width*board_height)
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):

        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act))

        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))

        return x_act, x_val
#data argument    
def get_equi_data(play_data):
    extend_data = []
    for state, mcts_porb, winner in play_data:
        for i in [1,2,3,4]:

            equi_state=np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob=np.rot90(np.flipud(mcts_porb.reshape(6, 6)),i)
            extend_data.append((equi_state,np.flipud(equi_mcts_prob).flatten(),winner))

            equi_state=np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob=np.fliplr(equi_mcts_prob)
            extend_data.append((equi_state,np.flipud(equi_mcts_prob).flatten(),winner))
    return extend_data


if __name__ == '__main__':

    #define the para
    learn_rate=2e-3
    lr_multiplier=1.0 
    init_model=None
    best_win_ratio=0
    pure_mcts_playout_num=1000

    #define the env
    board=Board(width=6,height=6,n_in_row=4)
    game=Game(board)
    data_buffer=deque(maxlen=10000)

    #define the net,opt,agent with net
    policy_value_net=Net(6,6).cuda()
    optimizer=optim.Adam(policy_value_net.parameters(),weight_decay=1e-4)
    if init_model!=None:
        policy_value_net.load_state_dict(torch.load(init_model, encoding='latin1'))
    mcts_player=MCTSPlayer(policy_value_net,c_puct=5,n_playout=400,is_selfplay=1)

    #start train
    for i in range(1500):
        #collect data and adta argument,play until one wins
        _,play_data=game.start_self_play(mcts_player,temp=1)
        play_data=list(play_data)[:]
        play_data=get_equi_data(play_data) #*8
        data_buffer.extend(play_data)
        if len(data_buffer)>512: 
            #update
            #-------------------------------------------------------------------------------------
            #load the data
            mini_batch=random.sample(data_buffer,512)
            state_batch=[data[0] for data in mini_batch]
            mcts_probs_batch=[data[1] for data in mini_batch]
            winner_batch=[data[2] for data in mini_batch]

            #array to tensor
            state_batch=Variable(torch.FloatTensor(np.array(state_batch)).cuda())
            mcts_probs=Variable(torch.FloatTensor(np.array(mcts_probs_batch)).cuda())
            winner_batch=Variable(torch.FloatTensor(np.array(winner_batch)).cuda())    
            
            #Train(get loss)
            for j in range(5):
                #caculate one batch loss to update net
                optimizer.zero_grad()
                for param_group in optimizer.param_groups: param_group['lr']=learn_rate*lr_multiplier
                log_act_probs,value=policy_value_net(state_batch)
                #we want the net output value is close to actual winner
                value_loss=F.mse_loss(value.view(-1),winner_batch)
                #we want the net output policy is close to mcts policy
                policy_loss=-torch.mean(torch.sum(mcts_probs*log_act_probs,1))
                #loss = (z - v)^2 - pi^T * log(p) 
                loss=value_loss+policy_loss
                loss.backward()
                optimizer.step()
                total_loss=loss.item()                
                
            #-------------------------------------------------------------------------------------
        if (i+1) % 50 == 0:  #50
            #Evaluate
            #-------------------------------------------------------------------------------------
            #load the train model
            current_mcts_player=MCTSPlayer(policy_value_net,c_puct=5,n_playout=400)
            pure_mcts_player=MCTS_Pure(c_puct=5,n_playout=pure_mcts_playout_num)
            win_cnt=defaultdict(int)
            #simulate a game
            for i in range(10):
                #change the first player,adapt to all situation
                winner=game.start_play(current_mcts_player,pure_mcts_player,start_player=i%2,is_shown=0)        
                win_cnt[winner]+=1

            #win plus 1,tie plus 0.5
            win_ratio=(win_cnt[1]+0.5*win_cnt[-1])/10
            print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                    pure_mcts_playout_num,
                    win_cnt[1], win_cnt[2], win_cnt[-1]))
            #-------------------------------------------------------------------------------------

            #save
            #-------------------------------------------------------------------------------------
            #policy_value_net.save_model('./current_policy.model')
            torch.save(policy_value_net.state_dict(),'./current_policy.model')
            if win_ratio>best_win_ratio:
                print("New best policy!!!!!!!!")
                best_win_ratio=win_ratio
                #policy_value_net.save_model('./best_policy.model')
                torch.save(policy_value_net.state_dict(),'./best_policy.model')

                #although AI win all the game,Maybe because the opponent is not strong enough, so you have to strengthen the opponent first
                if (best_win_ratio==1 and pure_mcts_playout_num<5000):
                    pure_mcts_playout_num+=1000
                    best_win_ratio=0
            #-------------------------------------------------------------------------------------
