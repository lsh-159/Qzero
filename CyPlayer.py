'''
YAI-quoridor/src/CyPlayer.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import math
import numpy as np

from Settings import *
from CyCy import PyBoard as Board
from CyCy import PyMCTS as MCTS
from CyCy import PyMCTS_V as MCTS_V
Board.init()

try:
    import faulthandler
    faulthandler.enable()
    print('Faulthandler enabled in CyPlayer')
except Exception:
    print('Could not enable faulthandler in CyPlayer')

    
class Player:

    def __init__(self, Verbose = False):

      ####### init Board, MCTS  ################
      self.game = Board()
      if HPARAMS['bandit'] == "PUCT" :       
          self.mcts = {0:MCTS(),1:MCTS()} 
      elif HPARAMS['bandit'] == 'impUCB' :  
          self.mcts = {0:MCTS_V(),1:MCTS_V()} 
      else :
          raise ValueError("Use PUCT or impUCB")

      ####### init per action variables ########
      self.currMCTS = 0        # 0 or 1 (not game.colour)
      self.currBOARD_CNT = 0
      self.playout_num = 0
      self.playout_max = PLAYOUT_FULL if random.random()<0.25 else PLAYOUT
      
      ####### init per game variables ##########
      self.data = {0:[],1:[]}
      self.first_r_moves = np.random.exponential(scale=4.0, size=None)
      if not HPARAMS['useNN'] :  # 0.5~4.5  1~15  0.4~1.5
          self.eval_a = np.random.uniform(low=0.5, high=4.5, size=2)
          self.eval_b = np.random.uniform(low=1.0, high=15.0, size=2)
          self.eval_c = np.random.uniform(low=0.4, high=1.5, size=2)
          self.p1 = [round(self.eval_a[0],3), round(self.eval_b[0],3), round(self.eval_c[0],3)]
          self.p2 = [round(self.eval_a[1],3), round(self.eval_b[1],3), round(self.eval_c[1],3)]
          
      ####### variables for outside  ###########
      self.player_no = TPARAMS['player_no']
      TPARAMS['player_no']+= 1
      self.printInterval = HPARAMS['printInterval']
      self.NNinp =   np.zeros((16,9,9),dtype=np.float32, order='C')

      ##### for Display Statistics ######
      self.Verbose = Verbose
      if self.Verbose :
          self.prepareStatistics()

    def prepareStatistics(self):
        """
        rootchild : rootchild()     # (action, N, Q*10^7, V*10^7, P*10^7 )
        pp_h : self.playout_num() history
        Narr_h : Narr history       # not use currently
        V_h : rootnodeV() history
        VAux_h : rootnodeS() history
        Q_h : rootnodeQ() history
        B_h, D_h, history : B, D, action history
        Qstd_h ?
        """
        StatKeys = ['rootchild', 'pp_h', 'Narr_h', 'V_h', 'VAux_h', 'Q_h', 'B_h', 'D_h', 'history']

        self.Stat = {}
        for key in StatKeys :
            self.Stat[key] =  {0: [], 1: [] } 

        self.Stat['history'] = []
        self.Stat['is_SelfPlay'] = True
        self.Stat['NNname'] = {0: TPARAMS['NNname'], 1: TPARAMS['NNname'] }
        if not HPARAMS['useNN'] :
            self.Stat['NNname'] = {0: f"Heuristic({self.p1})", 1: f"Heuristic({self.p2})" }

        pass  

    #So on my machine, filling vectors with less than 800 elements is faster than creating new vectors,  with more than 800 elements creating new vectors gets faster
    def toNNinput(self):
        self.NNinp.fill(0)
        self.game.toNNinput(self.NNinp)
        New_NNinp = np.expand_dims(self.NNinp , axis=0)
        New_NNinp  = torch.from_numpy(New_NNinp)
        return New_NNinp

    def playout(self, deterministic_ = False):
      colour = self.currMCTS

      if not deterministic_ :   # dirichlet noise
          if (self.playout_num == 1) and (self.currBOARD_CNT <= self.first_r_moves) : 
              if HPARAMS['bandit'] == "PUCT" :
                  self.mcts[self.currMCTS].makeNoise()
                  
      if HPARAMS['bandit'] == "PUCT" :
          self.mcts[colour].playout(self.game, forced_playout = not deterministic_) 
      elif HPARAMS['bandit'] == "impUCB" :
          self.mcts[colour].playout_V(self.game) 
      self.playout_num += 1
      return self.toNNinput()

    def step(self, Parr, v, Score_arr, deterministic_ = False):
        colour = self.currMCTS    
        expand_ = True
        score_ = 0  # self.game.scoreeval() or Calculate_mean(Score_arr)
        if self.game.gameend() :
            expand_ = False
            v = self.game.eval()
            score_ = self.game.score()
            
        if HPARAMS['bandit'] == "PUCT" :
            self.mcts[colour].backprop(self.game, Parr, v, score_, expand_)    
        elif HPARAMS['bandit']== "impUCB" :
            self.mcts[colour].backprop(self.game, v, score_, expand_)   
        
        # first r moves randomly with proportional to NN policy
        if (HPARAMS['useNN'] == True)  and (self.currBOARD_CNT <= self.first_r_moves) and (not deterministic_) :
            sumR, Rarr = self.choose_rawNNPolicy_with_ValidMoves(Parr)
            action = np.random.choice(len(np.array(Rarr)), p=Rarr)
            if self.Verbose :
                self.RecordStats(Narr, action)
            self.interaction(self.game, action)
            if self.game.gameend():
                self.dataPreprocess() # nothing happens. just logging to console

        # playout >= 1000, do action(interaction)
        if self.playout_num >= self.playout_max: 
            Narr = np.zeros((140,),dtype=np.float32, order='C')
            action = self.mcts[colour].getAction(Narr,deterministic_)
            #print(self.mcts[colour].pv(self.game))
            if self.Verbose :
                self.RecordStats(Narr, action)
            if self.playout_max==PLAYOUT_FULL:
                self.data[colour].append([self.toNNinput().clone().detach(), Narr, None , None, None])
            self.interaction(self.game, action)
            if self.game.gameend():
                self.dataPreprocess()
  
    def interaction(self, state, action):
        # environment인 Board와 MCTS rootnode 업데이트, MCTS 바꿔주기
        self.game.doAction(action)
        self.game.reset_path()
        if self.player_no % self.printInterval == 0 :
            print(f"Next action chosen// [player_no {self.player_no}] et al,\t[b_cnt :{self.currBOARD_CNT }->{self.game.get_count()}, pp :{self.playout_num}, act :{action}]",end=" ")#\t\tNarr : {Narr}")
            print("\t", self.game.get_history())
        self.mcts[0].update(action)
        self.mcts[1].update(action)
        self.playout_num = 0
        self.playout_max = PLAYOUT_FULL if random.random()<0.25 else PLAYOUT
        self.currMCTS = 1 - self.currMCTS
        self.currBOARD_CNT += 1
        pass
    
    def choose_rawNNPolicy_with_ValidMoves(self, Parr):
        # 네트워크 policy인 Parr 중 Invalid move는 0으로 패딩 후 normalize (sum이 1이 되게) -> Rarr
        # sum_R 은 normalize 직전 Rarr 값 sum.  1에 가까울수록 좋은 네트워크 (sanity check)
        # ValidMask는 np.array(140,) : valid moves 만 1, 나머지는 0 
        assert(self.playout_num == 1)    
        ValidMask = np.zeros(140)
        ValidActions = np.zeros(140, dtype=np.int32)
        self.game.getAvailableAction(ValidActions)
        ValidActions = set(ValidActions)
        for j in ValidActions :
            ValidMask[j] = 1                
        Rarr = Parr * ValidMask   # heuristic도 first r random move 하고싶으면  Rarr = ValidMask, 근데 필요없어보임
        sum_R = Rarr.sum()
        for Ract in ValidActions:
            Rarr[Ract] = Rarr[Ract]/ sum_R ;
        return sum_R, Rarr

    def RecordStats(self,Narr, action):
        #StatKeys = ['rootchild', 'pp_h', 'Narr_h', 'V_h', 'VAux_h', 'Q_h', 'B_h', 'D_h'.' history']
        
        self.Stat['rootchild'][self.currMCTS].append(self.mcts[self.currMCTS].rootchild()) # (action, N, Q*10^7, V*10^7, P*10^7 )
        self.Stat['pp_h'][self.currMCTS].append(self.playout_num)
        self.Stat['Narr_h'][self.currMCTS].append(Narr) 
        self.Stat['V_h'][self.currMCTS].append(self.mcts[self.currMCTS].rootnodeV()) 
        self.Stat['VAux_h'][self.currMCTS].append(self.mcts[self.currMCTS].rootnodeS())
        self.Stat['Q_h'][self.currMCTS].append(self.mcts[self.currMCTS].rootnodeQ()) 
        self.Stat['B_h'][self.currMCTS].append(self.game.get_b_remain() if self.currMCTS else self.game.get_a_remain())
        self.Stat['D_h'][self.currMCTS].append(self.game.get_b_path_len() if self.currMCTS else self.game.get_a_path_len()) 
        self.Stat['history'].append(action)         
        pass    

    def dataPreprocess(self):
        if self.game.get_winner() == 2 :   # is draw : 150수 초과  or  (40수이상 and 반복수)
            self.data[0] = []
            self.data[1] = []  # abort data 

        else :                            # is not draw.
            if any(self.data[0]) and any(self.data[1]) :
                print(f"collected len(data[0]):{len(self.data[0])},  len(data[1]):{len(self.data[1])}")
            for sample in self.data[0]:
                cnt = sample[0][0,14,0,0].int()          #game.count == even number 
                #color = sample[0][0,15,0,0].int()         #game.colour == 0 always
                if (cnt+1) == len(self.game.get_history()) :  
                    del self.data[0][-1]
                    break
                opp_action = self.game.get_history()[cnt + 1] 
                opp = [0 for _ in range(140)]
                opp[opp_action] = 1
                z = 1 if (self.game.get_winner() == 0) else -1
                sample[2] = opp
                sample[3] = z
                sample[4] = self.game.get_b_path_len() - self.game.get_a_path_len()
            for sample in self.data[1]:
                cnt = sample[0][0,14,0,0].int()            #game.count == odd number
                #color = sample[0][0,15,0,0].int()          #game.colour == 1 always
                if (cnt+1) == len(self.game.get_history()) : 
                    del self.data[1][-1]
                    break              
                opp_action = self.game.get_history()[cnt + 1] 
                opp = [0 for _ in range(140)]
                opp[opp_action] = 1
                z = 1 if (self.game.get_winner() == 1) else -1
                sample[2] = opp
                sample[3] = z
                sample[4] = self.game.get_a_path_len() - self.game.get_b_path_len()
