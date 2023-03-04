'''
YAI-quoridor/src/CyArena.py

Perform Gating between Two NNs & Record Statistics
Reference:
[1] https://github.com/suragnair/alpha-zero-general
'''
import random
import numpy as np
import torch
import shutil
import glob
import datetime 
from pytz import timezone
import time
import os
import re
from math import floor
from tqdm import tqdm

from Model import *
from Settings import *
from CyCy import PyBoard as Board
from CyCy import PyMCTS as MCTS
from CyCy import PyMCTS_V as MCTS_V
Board.init()

try:
    import faulthandler
    faulthandler.enable()
    print('Faulthandler enabled in Arena')
except Exception:
    print('Could not enable faulthandler in Arena')


class ArenaPlayer:

    def __init__(self, arena_player_no, Verbose = True):

      ####### init Board, MCTS  ################
      self.game = Board()
      self.mcts ={0:MCTS(),1:MCTS()} 

      ####### init per action variables ########
      self.currMCTS = 0        # 0 or 1 (not game.colour)
      self.currBOARD_CNT = 0      
      self.playout_num = 0
      self.playout_max = HPARAMS['Arena_PLAYOUT_N']

      ####### init per game variables ##########
      self.data = {0:[],1:[]}
      self.first_r_moves = np.random.exponential(scale=4.0, size=None)

      ####### variables for outside  ###########
      self.player_no = arena_player_no
      self.printInterval = 1
      
      ##### for Display Statistics ######
      self.Verbose = Verbose
      self.Verbose = Verbose if random.random()<HPARAMS['SaveStatisticsProbability'] else False
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
        self.Stat['is_SelfPlay'] = False
        self.Stat['NNname'] = {0: None, 1: None }
        pass  

    def toNNinput(self):
        NNinp = np.zeros((16,9,9),dtype=np.float32, order='C')
        self.game.toNNinput(NNinp)
        NNinp = np.expand_dims(NNinp, axis=0)
        return torch.from_numpy(NNinp)

    def playout(self, deterministic_ = True):

      colour = self.currMCTS
      if True :  # dirichlet noise
          if (self.playout_num == 1) and (self.currBOARD_CNT <= self.first_r_moves) : # first r moves randomize 
              self.mcts[self.currMCTS].makeNoise()

      self.mcts[colour].playout(self.game, forced_playout = not deterministic_) 
      self.playout_num += 1
      return self.toNNinput()

    def step(self, Parr, v, Score_arr,deterministic_ = True):
        colour = self.currMCTS    
        expand_ = True
        score_ = 0 # self.game.scoreeval()
        if self.game.gameend() :
            expand_ = False
            v = self.game.eval()      
            score_ = self.game.score()  

        # playout < 1000     
        self.mcts[colour].backprop(self.game,Parr, v, score_,expand_)    

        # playout >= 1000    
        if self.playout_num >= self.playout_max: 
            Narr = np.zeros((140,),dtype=np.float32, order='C')
            action = self.mcts[colour].getAction(Narr,deterministic_) 
            if self.Verbose :
                self.RecordStats(Narr,action)   
            self.interaction(self.game, action)

    def interaction(self, state, action):
        # environment인 보드와 MCTS 업데이트
        self.game.doAction(action)
        self.game.reset_path()
        if self.player_no % self.printInterval == 0 :  
            print(f"Arena >> Game Board{self.player_no} : [{['Red', 'Blue'][self.currMCTS]}]\tchoose action {action}!\t[b_cnt :{self.currBOARD_CNT }->{self.game.get_count()}, pp :{self.playout_num}]", end=" ")#,\t\tNarr : {Narr}")
            print("\t", self.game.get_history()  )
        self.mcts[0].update(action)
        self.mcts[1].update(action)
        self.playout_num = 0
        self.playout_max = HPARAMS['Arena_PLAYOUT_N']
        self.currMCTS = 1 - self.currMCTS
        self.currBOARD_CNT += 1
        pass


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

class Arena:

    def __init__(self):
      self.NN = {0: None, 1:None}
      self.NNname = {0 : None, 1 : None}

      self.oneWon = 0
      self.twoWon = 0
      self.draws = 0

    def Gating(self, num_games):
        """
        Executes one episode of a game for parallel (for many ArenaPlayers) 
        Returns:
                winner: player who won the game (0 if player1, 1 if player2)  no draw implementation yet
        """
        self.oneWon = 0
        self.twoWon = 0
        self.draws = 0 
        Verboseflag = HPARAMS['arena_with_Verbose']

        inference_total_time = 0
        half_games = int(num_games / 2)
        Left_games = half_games
        start_time = time.time()
        print("====================================")
        print("")
        print(f"Arena >> Game begins...  Red is first turn")
        print("")
        print(f"         Player1 : {self.NNname[0]}     (current Champion)")
        print(f"         Player2 : {self.NNname[1]}      (trained  model )")
        print("")
        print(f"      total {half_games*2} games           Red          Blue ")
        print(f"         first {half_games} games :   [Player1]     [Player2]")
        print(f"         last  {half_games} games :   [Player2]     [Player1]")
        print("--------------------------------------------------------------")
        print("")
        print(f"       Player1 : Red(start)   vs     Player2 : Blue")
        print("")

        Arena_Players = [ArenaPlayer(idx, Verbose = Verboseflag) for idx in range(half_games)]  
        for Aplayer in Arena_Players :
            if Aplayer.Verbose :
                Aplayer.Stat['NNname'] = {0: self.NNname[0], 1: self.NNname[1] }
        while Arena_Players :
            batch = [Aplayer.playout(deterministic_= True) for Aplayer in Arena_Players]
            batch = torch.cat(batch, dim=0).float().to(DEVICE)
            with torch.no_grad():
                delta_time = time.time()
                output1, output2 = self.NN[Arena_Players[0].currMCTS](batch)
                p, opp = output1
                v, score = output2
            inference_total_time += time.time()-delta_time
            Ps = [p[i].cpu().numpy() for i in range(len(Arena_Players))]
            Vs = [v[i].item() for i in range(len(Arena_Players))]
            Scrs = [score[i].cpu().numpy() for i in range(len(Arena_Players))]

            remove_indices = []
            for i, Aplayer in enumerate(Arena_Players):
                Aplayer.step(Ps[i],Vs[i],Scrs[i], deterministic_ =True)
      
                if Aplayer.game.gameend():
                    Left_games -= 1
                    print("==============gameend=============== ")
                    print(f"Arena >> Game Board{Aplayer.player_no} reach gameend.  winner is [{['a','b','draw'][Aplayer.game.get_winner()]}]")
                    print("------------------------------------")
                    elapsed_total_time = time.time() - start_time
                    print("")
                    print(f"elapsed total time : {floor(elapsed_total_time/60)} min {round(elapsed_total_time % 60, 2)} secs")
                    print(f"Left games : {Left_games}, p1 : {self.oneWon}wins, p2 : {self.twoWon}wins, draw :{self.draws} ")
                    print("====================================")
                    print("\n")
                    winner = Aplayer.game.get_winner() 
                    if winner == 0 :
                        self.oneWon +=1
                    elif winner == 1 :
                        self.twoWon +=1
                    else :
                        self.draws +=1
                    remove_indices.append(i)
                    if Aplayer.Verbose :
                        save_stat(Aplayer.Stat)

            if remove_indices :
                somelist = [NotEndPlayer for j, NotEndPlayer in enumerate(Arena_Players) if j not in remove_indices]
                Arena_Players = []
                Arena_Players = somelist

        print(f"Arena >> {half_games} Games finished.")  
        print(f"Arena >> Statistics : Results of Player1 (Red) vs Player2 (Blue)")      
        print(f"P1 won {self.oneWon} times, \nP2 won {self.twoWon} times, \nDraw {self.draws} times.")
        print("--------------------------------------------------------------")
        
        NN0, NN1 = self.NN[0], self.NN[1]
        self.NN[0], self.NN[1] = NN1, NN0
        NN0name, NN1name = self.NNname[0], self.NNname[1]
        self.NNname[0], self.NNname[1] = NN1name, NN0name        

        Left_games = half_games
        print("==============================================================")
        print("==============================================================")
        print("==============================================================")
        print(f"Turn change. Now    Player2 (Red)  VS   Player1 (Blue)")
        print("==============================================================")
        print("==============================================================")
        print("==============================================================")
        print("")
        print(f"Arena :  Game begins...  Red is first turn, Blue is second")
        print("--------------------------------------------------------------")
        print("")
        print(f"       Player1 : Blue     vs     Player2 : Red(start)")
        print("")
        Arena_Players = [ArenaPlayer(idx, Verbose = True) for idx in range(half_games)]  
        for Aplayer in Arena_Players :
            if Aplayer.Verbose :
                Aplayer.Stat['NNname'] = {0: self.NNname[0], 1: self.NNname[1] }        
        while Arena_Players :
            batch = [Aplayer.playout(deterministic_= True) for Aplayer in Arena_Players]
            batch = torch.cat(batch, dim=0).float().to(DEVICE)
            with torch.no_grad():
                delta_time = time.time()
                output1, output2 = self.NN[Arena_Players[0].currMCTS](batch)
                p, opp = output1
                v, score = output2
            inference_total_time += time.time()-delta_time
            Ps = [p[i].cpu().numpy() for i in range(len(Arena_Players))]
            Vs = [v[i].item() for i in range(len(Arena_Players))]
            Scrs = [score[i].cpu().numpy() for i in range(len(Arena_Players))]

            remove_indices = []
            for i, Aplayer in enumerate(Arena_Players):
                Aplayer.step(Ps[i],Vs[i],Scrs[i], deterministic_ =True)
                if Aplayer.game.gameend():
                    Left_games -= 1
                    print("==============gameend=============== ")
                    print(f"Arena >> Game Board{Aplayer.player_no} reach gameend.  winner is [{['a','b','draw'][Aplayer.game.get_winner()]}]")
                    print("------------------------------------")
                    elapsed_total_time = time.time() - start_time
                    print("")
                    print(f"elapsed total time : {floor(elapsed_total_time/60)} min {round(elapsed_total_time % 60, 2)} secs")
                    print(f"Left games : {Left_games}, p1 : {self.oneWon}wins, p2 : {self.twoWon}wins, draw :{self.draws} ")
                    print("====================================")
                    print("\n")
                    winner = Aplayer.game.get_winner() 
                    if winner == 0 :
                        self.twoWon +=1
                    elif winner == 1 :
                        self.oneWon +=1
                    else :
                        self.draws +=1
                    remove_indices.append(i)
                    if Aplayer.Verbose :
                        save_stat(Aplayer.Stat)         
            if remove_indices :
                somelist = [NotEndPlayer for j, NotEndPlayer in enumerate(Arena_Players) if j not in remove_indices]
                Arena_Players = []
                Arena_Players = somelist

        NN0, NN1 = self.NN[0], self.NN[1]
        self.NN[0], self.NN[1] = NN1, NN0
        NN0name, NN1name = self.NNname[0], self.NNname[1]
        self.NNname[0], self.NNname[1] = NN1name, NN0name           

        #### loggingFlag ###
        print("=========*********==========++++++++++==================*********==========++++++++++==========")
        print(f"Arena >> Finished Gating.         [Player1 = {self.NNname[0]}] vs [Player2 = {self.NNname[1]}]")
        print(f"Arena >> Player1 : {self.NNname[0]}     (current Champion)")
        print(f"Arena >> Player2 : {self.NNname[1]}     (trained  model )")
        print(f"Arena >> elapsed total time : {floor((time.time()-start_time)/60)} min {round((time.time()-start_time) % 60,3)} secs")
        print(f"Arena >> P1 won {self.oneWon} times,\nP2 won {self.twoWon} times,\nDraw {self.draws} times.  ")
        print("=========*********==========++++++++++==================*********==========++++++++++==========")
        ####################

        return winner


    def change_Champion(self):
        if not HPARAMS['CHANGE_CHAMPION']:
          return
        print(f"Arena >> [Champion : {self.NNname[0]}]   [Challenger : {self.NNname[1]}]")
        if self.twoWon >= self.oneWon :
            print("Arena >> Challenger wins Champion.  (Draw means also Challenger's win) Champion aborted. Now Challenger becomes the Champion.")
            for file in os.scandir("./models/Champion"):
                os.remove(file.path)
            shutil.copyfile(f"./models/GatingPlayer1/{self.NNname[0]}", f"./models/LoseModels/{self.NNname[0]}")    
            shutil.copyfile(f"./models/GatingPlayer2/{self.NNname[1]}", f"./models/Champion/{self.NNname[1]}")
        else :
            print("Arena >> Champion wins Challenger.  No change happens.")
            shutil.copyfile(f"./models/GatingPlayer2/{self.NNname[1]}", f"./models/LoseModels/{self.NNname[1]}")

    def selectModels_intoGF(self):
        """
        GatingPlayer1, GatingPlayer2 Folder에 모델 저장.
        Champion 폴더의 모델을 GatingPlayer1 폴더로 copy
        Models 폴더에서 가장 최신 모델을 GatingPlayer2 폴더로 copy
        """
        models_list = glob.glob('./models/Champion/*.pth')      # ['./models/dual_net[6,96]-0001.pth']
        assert(len(models_list)==1)
        Champion_model_path = models_list[0]
        Champion_model_name = os.path.basename(Champion_model_path)  # 파일명만 추출
        Recent_model_path = max(glob.glob('./models/*.pth')) 
        Recent_model_name = os.path.basename(Recent_model_path) 
        
        if os.path.exists("./models/GatingPlayer1"):
            for file in os.scandir("./models/GatingPlayer1"):
                os.remove(file.path)
        shutil.copyfile(Champion_model_path, f"./models/GatingPlayer1/{Champion_model_name}")

        if os.path.exists("./models/GatingPlayer2"):
            for file in os.scandir("./models/GatingPlayer2"):
                os.remove(file.path)
        shutil.copyfile(Recent_model_path, f"./models/GatingPlayer2/{Recent_model_name}")

        print("=========*********==========++++++++++==========")
        print("Arena >> Gating Models selected.. ")
        print(f"\t\tCurrent Champion  : {Champion_model_path} -> player1") 
        print(f"\t\tRecent Challenger : {Recent_model_path}   -> player2")

        pass

    def loadModels_fromGF(self):
        """
        GatingPlayer1, GatingPlayer2 Folder로부터 모델 로드 후 self.NN,  self.NNname 값 바꾸기. (대국 전 반드시 호출 필요)
        """
        p1_folder = glob.glob('./models/GatingPlayer1/*.pth') # ['./models/GatingPlayer1/dual_net[6,96]-0001.pth']
        p2_folder = glob.glob('./models/GatingPlayer2/*.pth') 
        p1_path = p1_folder[0]
        p2_path = p2_folder[0]
        assert(len(p1_folder)==1)
        assert(len(p2_folder)==1)
        _, B1, C1, ver1 = re.findall(r'\d+', p1_path)
        _, B2, C2, ver2 = re.findall(r'\d+', p2_path)
        self.NN[0] = getModel(B1,C1)
        self.NN[1] = getModel(B2,C2)
        self.NNname[0] = os.path.basename(p1_path)
        self.NNname[1] = os.path.basename(p2_path)
        if torch.cuda.is_available() :
            self.NN[0].load_state_dict( torch.load(p1_path)['model_state_dict'] )    
            self.NN[1].load_state_dict( torch.load(p2_path)['model_state_dict'] )  
        else :
            self.NN[0].load_state_dict( torch.load(p1_path, map_location=torch.device('cpu'))['model_state_dict'] )   
            self.NN[1].load_state_dict( torch.load(p2_path, map_location=torch.device('cpu'))['model_state_dict'] )
        self.NN[0].to('cuda' if torch.cuda.is_available() else 'cpu')
        self.NN[1].to('cuda' if torch.cuda.is_available() else 'cpu')
        self.NN[0].eval()
        self.NN[1].eval()

        print("=========*********==========++++++++++==========")
        print("Arena >> Gating Models were loaded to memory Successfully.")
        print(f"\t\tPlayer1 : {p1_path}")
        print(f"\t\tPlayer2 : {p2_path}")
        pass

    def Mutex_Arena_lock(self, folder = "./log") :
        text_file_path = f"{folder}/Mutex_Arena.txt"

        nowLog = datetime.datetime.now(timezone('Asia/Seoul'))     #<class 'datetime.datetime'> 2022-12-09 21:37:54.213080+09:00
        now_str = f"{nowLog.year}-{nowLog.month}-{nowLog.day} {nowLog.hour}:{nowLog.minute}:{nowLog.second}"  #2022-12-9 21:37:54"

        edited_lines = []
        edited_lines.append(f"{now_str}\n")               
        lock_achieved = False
        with open(text_file_path,'r') as f:
            lines = f.readlines()               # ["2022-12-9 21:37:54\n", "0\n" ]
            pastLog = lines[0].strip()
            lockLog = lines[1].strip()
            if '0' in lockLog:
                lock_achieved = True           
                edited_lines.append("1\n")
                print(f"Arena >> Lock Mutex_Arena Success, {now_str}")
            elif '1' in lockLog :
                nowLog_rec = datetime.datetime.strptime(now_str, '%Y-%m-%d %H:%M:%S')
                pastLog_rec = datetime.datetime.strptime(pastLog, '%Y-%m-%d %H:%M:%S')     
                diff = nowLog_rec - pastLog_rec
                print(f"Arena >> Achieve Mutex_Arena_lock Failed. Someone is already gating models for {diff}")
                if HPARAMS['ARENA_TIMEOUT'] < diff :
                    lock_achieved = True
                    edited_lines.append("1\n")
                    print(f"Arena >> Arena_lock TIMEOUT :: abort lock. Someone maybe dead. He was gating for {diff}")
        if lock_achieved == True :
            with open(text_file_path,'w') as f:
                f.writelines(edited_lines)
        return lock_achieved


    def Mutex_Arena_unlock(self,folder = "./log") :
        text_file_path = f"{folder}/Mutex_Arena.txt"

        nowLog = datetime.datetime.now(timezone('Asia/Seoul'))     #<class 'datetime.datetime'> 2022-12-09 21:37:54.213080+09:00
        now_str = f"{nowLog.year}-{nowLog.month}-{nowLog.day} {nowLog.hour}:{nowLog.minute}:{nowLog.second}"  #2022-12-9 21:37:54"

        edited_lines = [f'{now_str}\n','0\n']
        with open(text_file_path,'w') as f:
            f.writelines(edited_lines)
        print(f"Arena >> Unlock Mutex_Arena Success, {now_str}")

    def prepare_Models(self):
        ##
        if HPARAMS['AUTOSELECT']:
          self.selectModels_intoGF()   ## 모델 직접 지정하고 싶으면 이 줄 주석처리하고 GatingPlayer 폴더에 직접 넣기
        self.loadModels_fromGF()
        return

if __name__ == "__main__":
  import os
  print("Running CyArena.py in cwd :",os.getcwd(), end="\n\n")
  QzeroArena = Arena()
  if QzeroArena.Mutex_Arena_lock() :
      QzeroArena.prepare_Models()
      QzeroArena.Gating(num_games = HPARAMS['Arena_GAME_NUM'])
      QzeroArena.change_Champion()
  QzeroArena.Mutex_Arena_unlock()
    
