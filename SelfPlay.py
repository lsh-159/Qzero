'''
YAI-quoridor/SelfPlay.py
'''


#### 221127 모델 구조 변경 다시 훈련?
#### 221201 모델 구조 안으로 interpolate 옮겨야함

import os
import re
import sys
import glob
import time
import torch
import pickle
import random
import logging
import numpy as np
from torch import nn
from torch import optim
from math import floor
from multiprocessing import Process, Queue

# Solve RecursionError: maximum recursion depth exceeded in comparison
sys.setrecursionlimit(10**7)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"      # Select GPU device number

TPARAMS, HPARAMS = {}, {}
from Settings import *
from Model import *
from CyPlayer import * 
from CyTrainer import * 
from src.CyArena import * 
from CyCy import PyBoard as Board
from CyCy import PyMCTS as MCTS
from CyCy import PyMCTS_V as MCTS_V
Board.init()

try:
    import faulthandler
    faulthandler.enable()
    print('Faulthandler enabled in SelfPlay')
except Exception:
    print('Could not enable faulthandler in SelfPlay')


# def main():
TPARAMS['models'] = glob.glob('./models/Champion/*.pth')       # ./models = ['./models/dual_net[6,96]-0001.pth', './models/dual_net[6,96]-0002.pth',]
TPARAMS['B'], TPARAMS['C'], TPARAMS['model_ver'] =  re.findall(r'\d+',TPARAMS['models'][-1])         
TPARAMS['NNname'] = TPARAMS['models'][-1]

TPARAMS['dual_net'] = getModel(TPARAMS['B'],TPARAMS['C'])
TPARAMS['dual_net'].load_state_dict( torch.load(TPARAMS['models'][-1], map_location=torch.device(DEVICE))['model_state_dict'] )    
TPARAMS['dual_net'].to(DEVICE)
TPARAMS['dual_net'].eval()

TPARAMS['trainer'] = Trainer()
TPARAMS['optimizer'] = optim.SGD(TPARAMS['dual_net'].parameters(), \
                                                  lr=HPARAMS['LR'], momentum=HPARAMS['MOMENTUM_DECAY'], weight_decay=HPARAMS['C_L2'])

Ps_Trash = np.zeros(HPARAMS['INFERENCE_BATCH_NUM'])
Scrs_Trash = np.zeros(HPARAMS['INFERENCE_BATCH_NUM'])
Vs_Trash = np.zeros(HPARAMS['INFERENCE_BATCH_NUM'])

data_buffer = []  # Replay Buffer       
start_time = time.time()
Verbose_flag = HPARAMS['selfplay_with_Verbose']

inference_total_time = 0              
TPARAMS['games_count'] = 0        # total number of games played
TPARAMS['draws_count'] = 0
TPARAMS['players'] = [Player(Verbose=Verbose_flag) for i in range(HPARAMS['INFERENCE_BATCH_NUM'])]     # total number of Players parallel computing
print("===============================================================================================")
print(f"selfplay init.     model : dual_net[{TPARAMS['B']},{TPARAMS['C']}]-{TPARAMS['model_ver']},    {HPARAMS['INFERENCE_BATCH_NUM']} players,   {DEVICE} ")
print(f"                   useNN : {HPARAMS['useNN']}    bandit : {HPARAMS['bandit']}    printInterval : {HPARAMS['printInterval']}  ")
print("===============================================================================================")
while True :
    batch = [player.playout(deterministic_= False) for player in TPARAMS['players']]
    

    if HPARAMS['useNN'] :
        batch = torch.cat(batch, dim=0).float().to(DEVICE)
        with torch.no_grad():
            delta_time = time.time()
            output1, output2 = TPARAMS['dual_net'](batch)
            p, opp = output1
            v, score = output2
        inference_total_time += time.time()-delta_time
        Ps = [p[i].cpu().numpy() for i in range(HPARAMS['INFERENCE_BATCH_NUM'])]
        Vs = [v[i].item() for i in range(HPARAMS['INFERENCE_BATCH_NUM'])]
        Scrs = [score[i].cpu().numpy() for i in range(HPARAMS['INFERENCE_BATCH_NUM'])]

    else :
        Ps = Ps_Trash
        Vs = [player.game.handeval(player.eval_a[player.currMCTS], player.eval_b[player.currMCTS], player.eval_c[player.currMCTS])  for player in TPARAMS['players']]
        Scrs = Scrs_Trash

    for i, player in enumerate(TPARAMS['players']):
        player.step(Ps[i],Vs[i],Scrs[i], deterministic_ =False)
    
        if player.game.gameend():
            TPARAMS['games_count'] += 1 
            TPARAMS['draws_count'] += 1 if player.game.get_winner() == 2 else 0
            data_buffer += player.data[1]
            data_buffer += player.data[0]

            elapsed_total_time = time.time() - start_time
            Agent_inCPUtime = round(elapsed_total_time - inference_total_time, 2)
            avg_game_time = elapsed_total_time / TPARAMS['games_count']               
            print("\n\n==============gameend===============  gamecount = ",TPARAMS['games_count'], "  drawcount", TPARAMS['draws_count'] )
            print(f"CyPlayer >> Game_no {player.player_no} reach gameend.  winner is {['a','b','draw'][player.game.get_winner()]}, boardCNT = {player.currBOARD_CNT}")
            print(f"elapsed total time : {floor(elapsed_total_time/60)} min {round(elapsed_total_time) % 60} secs / {TPARAMS['games_count']} games   =  average {floor(avg_game_time/60)}min {round(avg_game_time%60,3)} sec per game")
            print(f"Agent CPU time : {floor(Agent_inCPUtime/60)} min {round(Agent_inCPUtime % 60,2)} secs")
            print(f"Agent GPU time : {floor(inference_total_time/60)} min {round(inference_total_time % 60,2)} secs")
            if data_buffer : 
                print(f"data saved --> Replay_Buffer now has {len(data_buffer)} samples, Replay_Buffer[-1] =  [{data_buffer[-1][0].shape}, {len(data_buffer[-1][1])}, {len(data_buffer[-1][2])}, {data_buffer[-1][3]}, {data_buffer[-1][4]}]")
            else :
                print(f"data_buffer : no data collected")
            print("history :", player.game.get_history())
            print("====================================")
            
            if Verbose_flag :
                save_stat(player.Stat)
            TPARAMS['players'][i] = Player(Verbose=Verbose_flag)


            # 2000 샘플마다 데이터저장
            if data_buffer and (len(data_buffer) >= HPARAMS['data_buffer_sz']) :       
                dataver = save_data(data_buffer, TPARAMS['B'],TPARAMS['C'], TPARAMS['model_ver'])
                print("\n\n\n\n")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")   
                print(f"==========================={HPARAMS['data_buffer_sz']} samples complete, saving data... ===================================")
                print(f"{TPARAMS['games_count']} games finished,  elapsed time :{floor((time.time() - start_time)/60)} min {round((time.time() - start_time)) % 60} secs")
                print(f"Sampled data saved.  data len : {len(data_buffer)}, data ver : {dataver}, model ver : {TPARAMS['model_ver']}")
                print("====================================================================================================")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")                  
                data_buffer.clear()  #del data_buffer[:]


                # 훈련 미사용(hit_num == 0) 데이터가 25만개(Window 개)이상  : 훈련 진행 
                if HPARAMS['selfplay_with_train']:
                  TPARAMS['trainer'].step()
                  print("\n\n\n\n")

                if HPARAMS['selfplay_with_arena'] and (int(TPARAMS['model_ver']) % HPARAMS['GATING_snapshots_no'] ==0) : #업데이트 4개마다 gating
                    QzeroArena = Arena()
                    if QzeroArena.Mutex_Arena_lock() :
                        QzeroArena.prepare_Models()
                        QzeroArena.Gating(num_games = HPARAMS['Arena_GAME_NUM'])
                        QzeroArena.change_Champion()
                    QzeroArena.Mutex_Arena_unlock()

                if HPARAMS['useNN'] :
                    # 2000 샘플마다 데이터 최신 Champion 모델로 변경
                    TPARAMS['dual_net'], TPARAMS['B'], TPARAMS['C'], TPARAMS['model_ver'] = TPARAMS['trainer'].load_champion(folder = "./models/Champion", filename = None) 
                    TPARAMS['NNname'] = f"./models/dual_net[{TPARAMS['B']},{TPARAMS['C']}]-{TPARAMS['model_ver']}.pth"
                    TPARAMS['dual_net'].to(DEVICE)
                    TPARAMS['dual_net'].eval()

# if __name__ == "__main__":
#     main()

