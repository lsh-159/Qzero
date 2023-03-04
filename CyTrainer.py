'''
YAI-quoridor/src/CyTrainer.py
'''
import matplotlib.pyplot as plt
import torch
import torchvision
import pickle
import glob
import os
import logging
import re
import time
import datetime
import random
from pytz import timezone
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from math import floor
from math import ceil
from tqdm import tqdm
#from torch.optim.swa_utils import AveragedModel, SWALR
#from torchcontrib.optim import SWA
from Settings import *
from Model import * 

BLOCK_ARR = [idx for idx in range(0,128)]
MOVE_ARR = [idx for idx in range(128,140)]
VERTICAL_FILP_BLOCK_ARR = []
VERTICAL_FILP_MOVE_ARR = [131,129,130,128,   135,133,134,132,  138,139,136,137]
HORIZONTAL_FILP_BLOCK_ARR = []
HORIZONTAL_FILP_MOVE_ARR =  [128,130,129,131,   132,134,133,135 ,  137,136,139,138]
for row in range(0,8):
    VERTICAL_FILP_BLOCK_ARR += [idx for idx in range((7-row)*8,(8-row)*8)]
for row in range(8,16):
    VERTICAL_FILP_BLOCK_ARR +=  [idx for idx in range((15+8-row)*8,(16+8-row)*8)]
for row in range(16):   
    HORIZONTAL_FILP_BLOCK_ARR +=[idx for idx in range((row+1)*8 -1, row*8 -1, -1)]


class Board_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root = './data',  hparameters= None, transforms_=None):

        print("")
        print("")
        print("=========================================================================")
        print(f"board_dataset __init__ : checking {data_root}")
        print("=========================================================================")
        self.root_folder = glob.glob(data_root + '/*.pkl')     # ['./data/package[2004,b5c64,6000,2,0]-[2022-12-28_22h-28m-11s-142152].pkl',  ... ]
        self.PackageIndexes = set()
        self.ReqEpoches = None        
        Nwin, ReqH = get_Nwindow_RequiredHit(folder = data_root)
        self.Nwindow = Nwin
        self.ReqHit = ReqH

        ########## 필요 hit 수 > 현재 hit수 인 Package Index 를 self.PacakgeIndexes에 저장 ########
        print(f"Train Dataset >> Detecting Hit-not sufficient data ...",end = "\t\t") 
        if len(self.root_folder) != 0 :
            num_samples_sum = 0
            for Packidx in range(len(self.root_folder)):
                info = re.findall(r'\d+',self.root_folder[Packidx])
                samples_n, model_b, model_c, pp, Required_hit, Curr_hit =   info[0],  info[1],  info[2],  info[3],  info[4],  info[5]

                if int(Required_hit) > int(Curr_hit) :
                    self.PackageIndexes.add(Packidx)
                    num_samples_sum += int(samples_n)
        print(f"complete. Detected {len(self.PackageIndexes)} packs ({num_samples_sum} samples)")
        
        self.ReqEpoches = ceil(num_samples_sum/HPARAMS['LoadMemoryPartition'])
        if self.ReqEpoches != 1 :
            print(f"Train Dataset >> Since {num_samples_sum} samples >= {HPARAMS['LoadMemoryPartition']},  total {self.ReqEpoches} epoches will be processed")
        else :
            print(f"Train Dataset >> Since {num_samples_sum} samples <= {HPARAMS['LoadMemoryPartition']},  total {self.ReqEpoches} epoches will be processed") 
        ###########################################################################################



        ############  로드할 Package 불러오기 (30만개(HPARAMS['LoadMemoryPartition']) 샘플씩 분할해서) ########
        self.Packindexlist = sorted(self.PackageIndexes)  # 10,14,150,432,433,434,... 리스트로 변경
        random.shuffle(self.Packindexlist )         # 최초 1회만 shuffle.
        self.CurrEpoch = 0
        self.Count = 0
        self.Samples_buffer = []       #./data에서 pacakge를 로드해 샘플(s,pi,opp,v,sco)단위로 저장
        self.NumSamplesInMemory = 0
        self.LoadData_ToMemory()       # HPARAMS['LoadMemoryPartition'] 만큼만
        ####################################################################################################### 

    def LoadData_ToMemory(self):
        """
        
        """

        self.Packcnt = 0
        self.NumSamplesInMemory = 0
        self.Samples_buffer = []    
        self.memory_s = []
        self.memory_pi = []
        self.memory_opp = []
        self.memory_z = []
        self.memory_score = []            
        ####################### Load data to Memory ##########################
        print(f"Train Dataset >> Loading Packages for Sub Epoch {self.CurrEpoch + 1} (less than {HPARAMS['LoadMemoryPartition']} samples for memory issue)..")
        if self.Count >= len(self.Packindexlist) :
            print("Logic Error : this function should not be called, so many called.")
            return 0
        while self.Count < len(self.Packindexlist) :
            Packidx = self.Packindexlist[self.Count]
            info = re.findall(r'\d+',self.root_folder[Packidx])
            samples_n = info[0]

            if (self.NumSamplesInMemory + int(samples_n)) <= HPARAMS['LoadMemoryPartition'] :
                with open(self.root_folder[Packidx], 'rb') as f : 
                    data = pickle.load(f) 
                    self.Samples_buffer += data
                    self.Packcnt +=1
                self.NumSamplesInMemory += int(samples_n)   
                self.Count += 1            
            else :
                break
        self.CurrEpoch += 1    
        for idx, sample in enumerate(self.Samples_buffer) :  #여기 for문은 1초 내로 걸림
            self.memory_s.append(sample[0])     # torch.Size([1, 16, 9, 9])
            self.memory_pi.append(sample[1])    # <class 'numpy.ndarray'>
            self.memory_opp.append(sample[2])   # <class 'list'>
            self.memory_z.append(sample[3])     # int 1 or -1
            self.memory_score.append(sample[4]) # int  
        self.Samples_buffer = []
        print(f"Train Dataset >> Successfully loaded {self.Packcnt} packs ({self.NumSamplesInMemory} samples) to memory...for Sub Epoch {self.CurrEpoch}")
        self.viewDataHistogram(saveas = "./log/Histograms/DataHistogram")
        ########################################################################


        ############  Data Augmentation (Vertical, Horizontal)  ################
        if HPARAMS['use_Augmentation'] :
            prev_len = len(self.memory_s)
            st_time = time.time()
            if DEVICE == 'cuda'  or DEVICE == 'cpu':
                print(f"Train Dataset >> Starting Data augmentation(batch) in {DEVICE},\tCurrent N_samples = ", len(self.memory_s))
                self.batch_dataAugmentation()
            # elif DEVICE == 'cpu' :
            #     print("Train Dataset >> Starting Data augmentation in CPU,\tCurrent N_samples = ", len(self.memory_s))
            #     self.dataAugmentation()
            else :
                print("Logic Error")

            elap_time = time.time() - st_time
            print("Train Dataset >> End Data augmentation,     \tCurrent N_samples = ", len(self.memory_s),f"  elapsed time : {floor((elap_time)/60)} min {round(elap_time % 60,2)} secs , Making {round((len(self.memory_s)-prev_len)/round(elap_time),2)} samples per sec ") 
            self.viewDataHistogram(saveas = "./log/Histograms/DataHistogram(Aug)")     
        ######################################################################## 

        return 1   

    def __getitem__(self, index):
        # memory_s[index] : torch.Size([1, 16, 9, 9])
        # memory_pi[index] : <class 'numpy.ndarray'> or torch.Tensor(if batch-augmented)
        # memory_opp[index] : <class 'list'> or torch.Tensor(if batch-augmented)
        # memory_z[index] : int 1 or -1
        # memory_score[index] :int 

        s = self.memory_s[index]      
        squez_s = s.squeeze(0)

        z = self.memory_z[index]      
        one_hot_z = 1 if z == 1 else 0

        score = self.memory_score[index]  
        if score > 20 :
            score = 20
        elif score < -20 :
            score = -20
        score += 20
        one_hot_score = torch.zeros(41)
        one_hot_score[score] = 1

        return {'s' : squez_s, 
                'p' : torch.Tensor(self.memory_pi[index]),         
                'opp' : torch.Tensor(self.memory_opp[index]),     
                'z' : one_hot_z,
                'score' : one_hot_score}  

    def __len__(self):
        return len(self.memory_s)


    def hit_cnt_update(self, folder = "./data"):
        for count in self.PackageIndexes :
            old_name = self.root_folder[count]  #'./data/package-00002[1500,b6c96,0].pkl'
            di = re.findall(r'\d+',old_name)
            pack_info = f"[{di[0]},b{di[1]}c{di[2]},{di[3]},{di[4]},{int(di[5])+1}]"
            date_info = f"[{di[6]}-{di[7]}-{di[8]}_{di[9]}h-{di[10]}m-{di[11]}s-{di[12]}]"

            new_name = f'{folder}/package{pack_info}-{date_info}.pkl'
            os.rename(old_name, new_name)
            time.sleep(0.1)
        pass

    def dataAugmentation(self):

        # info = {0 : "선수 플레이어 위치", 
        #         1 : "후수 플레이어 위치",
        #         2 : "가로 블럭 위치",
        #         3 : "세로 블럭 위치",
        #         4 : "(이전턴) 선수 플레이어 위치",
        #         5 : "(이전턴) 후수 플레이어 위치",
        #         6 : "(이전턴) 가로 블럭 위치",
        #         7 : "(이전턴) 세로 블럭 위치",
        #         8 : "a remain",
        #         9 : "b remain",
        #         10 : "(이전턴) a remain",
        #         11 : "(이전턴) b remain",
        #         12 : "a_path_len",
        #         13 : "b_path_len",
        #         14 : "board cnt",
        #         15 : "colour"}
        ss =[]
        pp =[]
        op =[]
        vv =[]
        sc =[]
        with torch.no_grad() :
            
            # X 2
            print("VerticalFlip_Augmentation")
            nums = np.random.choice([0, 1], size=len(self.memory_s), p=[.5, .5])
            for idx, state in enumerate(tqdm(self.memory_s)) :    
                state =   self.memory_s[idx]      # torch.Size([1, 16, 9, 9])
                policy =  self.memory_pi[idx]     # <class 'numpy.ndarray'>
                opp_policy = self.memory_opp[idx] # <class 'list'>
                value =   self.memory_z[idx]      # int 1 or -1
                score =   self.memory_score[idx]  # int

                if HPARAMS['Augment_randomly'] and nums[idx] :
                    continue

                Vstate, Vpi, Vopp, Vz, Vsc = self.VerticalFlip_Augmentation(state, policy, opp_policy, value, score)

                if not HPARAMS['Augment_randomly'] :
                    ss.append(Vstate)     
                    pp.append(Vpi)    
                    op.append(Vopp)   
                    vv.append(Vz)     
                    sc.append(Vsc) 
                else :
                    if nums[idx] :
                        self.memory_s[idx] = Vstate
                        self.memory_pi[idx] = Vpi
                        self.memory_opp[idx] = Vopp
                        self.memory_z[idx] = Vz
                        self.memory_score[idx] = Vsc

            self.memory_s += ss
            self.memory_pi += pp
            self.memory_opp += op 
            self.memory_z += vv
            self.memory_score += sc
            ss =[]
            pp =[]
            op =[]
            vv =[]
            sc =[]

            # X 4
            print("HorizontalFlip_Augmentation")
            nums = np.random.choice([0, 1], size=len(self.memory_s), p=[.5, .5])
            for idx, state in enumerate(tqdm(self.memory_s)) :     
                state =   self.memory_s[idx]      # torch.Size([1, 16, 9, 9])
                policy =  self.memory_pi[idx]     # <class 'numpy.ndarray'>
                opp_policy = self.memory_opp[idx] # <class 'list'>
                value =   self.memory_z[idx]      # int 1 or -1
                score =   self.memory_score[idx]  # int

                Hstate, Hpi, Hopp, Hz, Hsc = self.HorizontalFlip_Augmentation(state, policy, opp_policy, value, score)

                if not HPARAMS['Augment_randomly'] :
                    ss.append(Hstate)     
                    pp.append(Hpi)    
                    op.append(Hopp)   
                    vv.append(Hz)     
                    sc.append(Hsc) 
                else :
                    if nums[idx] :
                        self.memory_s[idx] = Hstate
                        self.memory_pi[idx] = Hpi
                        self.memory_opp[idx] = Hopp
                        self.memory_z[idx] = Hz
                        self.memory_score[idx] = Hsc

            self.memory_s += ss
            self.memory_pi += pp
            self.memory_opp += op 
            self.memory_z += vv
            self.memory_score += sc    
        
    def batch_dataAugmentation(self):

        assert(HPARAMS['Augment_randomly'])  # not random augment 아직 구현 X

        rand_idx_table = np.arange(len(self.memory_s))
        np.random.shuffle(rand_idx_table)
        half_size = round(len(self.memory_s)/2)
        with torch.no_grad():
          
            idx_table = rand_idx_table[:half_size]
            batch_num = floor(half_size/HPARAMS['BATCH_SIZE'])
            PrevIndex = 0
            CurrIndex = 0
            print("VerticalFlip_Augmentation (batch)")
            for batch_idx in tqdm(range(batch_num)):
                batch_s = []
                batch_pi = []
                batch_opp = []
                while( (CurrIndex - PrevIndex) != HPARAMS['BATCH_SIZE']) :
                    idx = idx_table[CurrIndex]
                    batch_s.append(self.memory_s[idx]  )                        # torch.Size([1, 16, 9, 9])
                    batch_pi.append(torch.Tensor(self.memory_pi[idx]).unsqueeze(0) )     # <class 'numpy.ndarray'>
                    batch_opp.append(torch.Tensor(self.memory_opp[idx]).unsqueeze(0)   )     # <class 'list'>
                    CurrIndex +=1  
                batch_s = torch.cat(batch_s, dim=0).to(DEVICE)
                batch_pi = torch.cat(batch_pi, dim=0).to(DEVICE)
                batch_opp = torch.cat(batch_opp, dim=0).to(DEVICE)

                batch_s, batch_pi, batch_opp = self.batch_VerticalFlip_Augmentation(batch_s, batch_pi, batch_opp)

                b_cnt = 0 
                while( (CurrIndex - PrevIndex) != 0) :
                    idx = idx_table[PrevIndex]
                    self.memory_s[idx] = batch_s[b_cnt:b_cnt+1,:,:,:].to('cpu')
                    self.memory_pi[idx] = batch_pi[b_cnt].to('cpu')
                    self.memory_opp[idx] = batch_opp[b_cnt].to('cpu')
                    PrevIndex +=1   
                    b_cnt += 1             
          
            rand_idx_table = np.arange(len(self.memory_s))
            np.random.shuffle(rand_idx_table)
            idx_table = rand_idx_table[:half_size]
            batch_num = floor(half_size/HPARAMS['BATCH_SIZE'])
            PrevIndex = 0
            CurrIndex = 0
            print("HorizontalFlip_Augmentation (batch)")
            for batch_idx in tqdm(range(batch_num)):
                batch_s = []
                batch_pi = []
                batch_opp = []
                while( (CurrIndex - PrevIndex) != HPARAMS['BATCH_SIZE']) :
                    idx = idx_table[CurrIndex]
                    batch_s.append(self.memory_s[idx]  )                        # torch.Size([1, 16, 9, 9])
                    batch_pi.append(torch.Tensor(self.memory_pi[idx]).unsqueeze(0) )     # <class 'numpy.ndarray'>
                    batch_opp.append(torch.Tensor(self.memory_opp[idx]).unsqueeze(0)  )     # <class 'list'>
                    CurrIndex +=1  
                batch_s = torch.cat(batch_s, dim=0).to(DEVICE)
                batch_pi = torch.cat(batch_pi, dim=0).to(DEVICE)
                batch_opp = torch.cat(batch_opp, dim=0).to(DEVICE)

                batch_s, batch_pi, batch_opp = self.batch_HorizontalFlip_Augmentation(batch_s, batch_pi, batch_opp)

                b_cnt = 0 
                while( (CurrIndex - PrevIndex) != 0) :
                    idx = idx_table[PrevIndex]
                    self.memory_s[idx] = batch_s[b_cnt:b_cnt+1,:,:,:].to('cpu')
                    self.memory_pi[idx] = batch_pi[b_cnt].to('cpu')
                    self.memory_opp[idx] = batch_opp[b_cnt].to('cpu')
                    PrevIndex +=1   
                    b_cnt += 1             

        pass

    def batch_HorizontalFlip_Augmentation(self, states, policys, opp_policys ):
        # value, score 그대로

        aug_states = states.clone().detach()

        # 기하학적 관련된 요소만 다 뒤집기
        aug_states[:,0:2,:,:] = torchvision.transforms.functional.hflip(img = aug_states[:,0:2,:,:])
        aug_states[:,2:4,:8,:8] = torchvision.transforms.functional.hflip(img = aug_states[:,2:4,:8,:8]) 
        aug_states[:,4:6,:,:] = torchvision.transforms.functional.hflip(img = aug_states[:,4:6,:,:]) 
        aug_states[:,6:8,:8,:8] = torchvision.transforms.functional.hflip(img = aug_states[:,6:8,:8,:8]) 

        ## 좌우 대칭
        #         print(f" ↑:{Ns[128]},  →:{Ns[129]},  ←:{Ns[130]},  ↓:{Ns[131]}")   
        #         print(f"2↑:{Ns[132]}, 2→:{Ns[133]}, 2←:{Ns[134]}, 2↓:{Ns[135]}")
        #         print(f"↗:{Ns[136]}, ↖:{Ns[137]}, ↘:{Ns[138]}, ↙:{Ns[139]}")
        #         print()

        # policy
        aug_policys = torch.zeros_like(policys)
        aug_policys[:,BLOCK_ARR] = policys[:,HORIZONTAL_FILP_BLOCK_ARR]
        aug_policys[:,MOVE_ARR] = policys[:,HORIZONTAL_FILP_MOVE_ARR]
        
        # opp policy
        aug_opp_policys = torch.zeros_like(opp_policys)
        aug_opp_policys[:,BLOCK_ARR] = opp_policys[:,HORIZONTAL_FILP_BLOCK_ARR]
        aug_opp_policys[:,MOVE_ARR] = opp_policys[:,HORIZONTAL_FILP_MOVE_ARR]

        return aug_states, aug_policys, aug_opp_policys

    def batch_VerticalFlip_Augmentation(self, states, policys, opp_policys):
        # value, score 그대로

        # 기하학적 관련된 요소 다 뒤집기
        aug_states = states.clone().detach()
        aug_states[:,0:2,:,:] = torchvision.transforms.functional.vflip(img = aug_states[:,0:2,:,:])
        aug_states[:,2:4,:8,:8] = torchvision.transforms.functional.vflip(img = aug_states[:,2:4,:8,:8]) 
        aug_states[:,4:6,:,:] = torchvision.transforms.functional.vflip(img = aug_states[:,4:6,:,:]) 
        aug_states[:,6:8,:8,:8] = torchvision.transforms.functional.vflip(img = aug_states[:,6:8,:8,:8]) 

        # 채널 위치교환으로 색깔 서로 바꾸기
        a_location = aug_states[:,0:1,:,:].clone()
        aug_states[:,0:1,:,:] = aug_states[:,1:2,:,:]
        aug_states[:,1:2,:,:] = a_location
        past_a_location = aug_states[:,4:5,:,:].clone()
        aug_states[:,4:5,:,:] = aug_states[:,5:6,:,:]
        aug_states[:,5:6,:,:] = past_a_location

        # 벽개수 서로 바꾸기
        a_remain = aug_states[:,8:9,:,:].clone()
        b_remain = aug_states[:,9:10,:,:]
        aug_states[:,8:9,:,:] = b_remain
        aug_states[:,9:10,:,:] = a_remain
        past_a_remain = aug_states[:,10:11,:,:].clone()
        past_b_remain = aug_states[:,11:12,:,:]
        aug_states[:,10:11,:,:] = past_b_remain
        aug_states[:,11:12,:,:] = past_a_remain

        # Pathlen도 서로 바꾸기
        a_path_len = aug_states[:,12:13,:,:].clone()
        b_path_len = aug_states[:,13:14,:,:]
        aug_states[:,12:13,:,:] = b_path_len
        aug_states[:,13:14,:,:] = a_path_len

        # board.color 명시적으로 바꾸기
        aug_states[:,15:16,:,:]  = 1- aug_states[:,15:16,:,:].clone()

        # policy
        aug_policys = torch.zeros_like(policys)
        aug_policys[:,BLOCK_ARR] = policys[:,VERTICAL_FILP_BLOCK_ARR]
        aug_policys[:,MOVE_ARR] = policys[:,VERTICAL_FILP_MOVE_ARR]
        
        # opp policy
        aug_opp_policys = torch.zeros_like(opp_policys)
        aug_opp_policys[:,BLOCK_ARR] = opp_policys[:,VERTICAL_FILP_BLOCK_ARR]
        aug_opp_policys[:,MOVE_ARR] = opp_policys[:,VERTICAL_FILP_MOVE_ARR]


        return aug_states, aug_policys, aug_opp_policys

    def HorizontalFlip_Augmentation(self, state, policy, opp_policy, value, score):
        # value, score 그대로

        aug_state = state.clone().detach()

        # 기하학적 관련된 요소만 다 뒤집기
        aug_state[0,0:2,:,:] = torchvision.transforms.functional.hflip(img = aug_state[0,0:2,:,:])
        aug_state[0,2:4,:8,:8] = torchvision.transforms.functional.hflip(img = aug_state[0,2:4,:8,:8]) 
        aug_state[0,4:6,:,:] = torchvision.transforms.functional.hflip(img = aug_state[0,4:6,:,:]) 
        aug_state[0,6:8,:8,:8] = torchvision.transforms.functional.hflip(img = aug_state[0,6:8,:8,:8]) 

        ## 좌우 대칭
        #         print(f" ↑:{Ns[128]},  →:{Ns[129]},  ←:{Ns[130]},  ↓:{Ns[131]}")   
        #         print(f"2↑:{Ns[132]}, 2→:{Ns[133]}, 2←:{Ns[134]}, 2↓:{Ns[135]}")
        #         print(f"↗:{Ns[136]}, ↖:{Ns[137]}, ↘:{Ns[138]}, ↙:{Ns[139]}")
        #         print()

        # policy
        aug_policy = np.zeros(140)
        aug_policy[BLOCK_ARR] = policy[HORIZONTAL_FILP_BLOCK_ARR]
        aug_policy[MOVE_ARR] = policy[HORIZONTAL_FILP_MOVE_ARR]
        
        # opp policy
        aug_opp_policy = np.zeros(140)
        opp_pi = np.array(opp_policy,dtype =int)
        aug_opp_policy[BLOCK_ARR] = opp_pi[HORIZONTAL_FILP_BLOCK_ARR]
        aug_opp_policy[MOVE_ARR] = opp_pi[HORIZONTAL_FILP_MOVE_ARR]

        aug_value = value
        aug_score = score
        return aug_state, aug_policy, aug_opp_policy, aug_value, aug_score
    
    def VerticalFlip_Augmentation(self, state, policy, opp_policy, value, score):
        # value, score 그대로

        # 기하학적 관련된 요소 다 뒤집기
        aug_state = state.clone().detach()
        aug_state[0,0:2,:,:] = torchvision.transforms.functional.vflip(img = aug_state[0,0:2,:,:])
        aug_state[0,2:4,:8,:8] = torchvision.transforms.functional.vflip(img = aug_state[0,2:4,:8,:8]) 
        aug_state[0,4:6,:,:] = torchvision.transforms.functional.vflip(img = aug_state[0,4:6,:,:]) 
        aug_state[0,6:8,:8,:8] = torchvision.transforms.functional.vflip(img = aug_state[0,6:8,:8,:8]) 

        # 채널 위치교환으로 색깔 서로 바꾸기
        a_location = aug_state[0,0:1,:,:].clone()
        aug_state[0,0:1,:,:] = aug_state[0,1:2,:,:]
        aug_state[0,1:2,:,:] = a_location
        past_a_location = aug_state[0,4:5,:,:].clone()
        aug_state[0,4:5,:,:] = aug_state[0,5:6,:,:]
        aug_state[0,5:6,:,:] = past_a_location

        # 벽개수 서로 바꾸기
        a_remain = aug_state[0,8:9,0,0].clone()
        b_remain = aug_state[0,9:10,0,0]
        aug_state[0,8:9,:,:] = b_remain
        aug_state[0,9:10,:,:] = a_remain
        past_a_remain = aug_state[0,10:11,0,0].clone()
        past_b_remain = aug_state[0,11:12,0,0]
        aug_state[0,10:11,:,:] = past_b_remain
        aug_state[0,11:12,:,:] = past_a_remain

        # Pathlen도 서로 바꾸기
        a_path_len = aug_state[0,12:13,0,0].clone()
        b_path_len = aug_state[0,13:14,0,0]
        aug_state[0,12:13,:,:] = b_path_len
        aug_state[0,13:14,:,:] = a_path_len

        # board.color 명시적으로 바꾸기
        aug_state[0,15:16,:,:]  = 1- aug_state[0,15,0,0]

        # policy
        aug_policy = np.zeros(140)
        aug_policy[BLOCK_ARR] = policy[VERTICAL_FILP_BLOCK_ARR]
        aug_policy[MOVE_ARR] = policy[VERTICAL_FILP_MOVE_ARR]
        
        # opp policy
        aug_opp_policy = np.zeros(140)
        opp_pi = np.array(opp_policy,dtype =int)
        aug_opp_policy[BLOCK_ARR] = opp_pi[VERTICAL_FILP_BLOCK_ARR]
        aug_opp_policy[MOVE_ARR] = opp_pi[VERTICAL_FILP_MOVE_ARR]

        aug_value = value
        aug_score = score
        return aug_state, aug_policy, aug_opp_policy, aug_value, aug_score


    def Rotation180_board(self, state, policy, opp_policy, value, score):
        # 그냥 horizontal filp + vertical filp 으로 구현 필요 x

        # 기하학적 관련된 요소 다 뒤집기
        # aug_state = state.clone().detach()
        # aug_state[0,0:1,:,:] = torchvision.transforms.functional.rotate(img = aug_state[0,0:1,:,:], angle = 180)
        # aug_state[0,1:2,:,:] = torchvision.transforms.functional.rotate(img = aug_state[0,1:2,:,:], angle = 180)
        # aug_state[0,2:3,:8,:8] = torchvision.transforms.functional.rotate(img = aug_state[0,2:3,:8,:8], angle = 180)
        # aug_state[0,3:4,:8,:8] = torchvision.transforms.functional.rotate(img = aug_state[0,3:4,:8,:8], angle = 180)
        # aug_state[0,4:5,:,:] = torchvision.transforms.functional.rotate(img = aug_state[0,4:5,:,:], angle = 180) 
        # aug_state[0,5:6,:,:] = torchvision.transforms.functional.rotate(img = aug_state[0,5:6,:,:], angle = 180)
        # aug_state[0,6:7,:8,:8] = torchvision.transforms.functional.rotate(img = aug_state[0,6:7,:8,:8], angle = 180)
        # aug_state[0,7:8,:8,:8] = torchvision.transforms.functional.rotate(img = aug_state[0,7:8,:8,:8], angle = 180)
        pass

    def viewDataHistogram(self, saveas = "./log/Histograms/DataHistogram"):

        print("Train Dataset >> Making Data histogram, \t Current N_samples = ", len(self.memory_s), end="\t\t")
        # 데이터의 state(z, score, s.cnt, s.colour, [s.a_remain,b_remain], [s.a_path_len, b_path_len] )    분포 그리기
        list_z = []
        list_score = []
        list_cnt = []
        list_color = []
        list_B_a = []
        list_B_b = []
        list_D_a = []
        list_D_b = []
        list_Narr = []

        for idx, state in enumerate(self.memory_s) : 

            list_z.append(self.memory_z[idx])
            list_score.append(self.memory_score[idx])                
            list_B_a.append(self.memory_s[idx][0][8][0][0])
            list_B_b.append(self.memory_s[idx][0][9][0][0])
            list_D_a.append(self.memory_s[idx][0][12][0][0])
            list_D_b.append(self.memory_s[idx][0][13][0][0])     
            list_cnt.append(self.memory_s[idx][0][14][0][0])
            list_color.append(self.memory_s[idx][0][15][0][0])       
            list_Narr.append(np.array(self.memory_pi[idx]))            
            #self.memory_pi[idx]
            #self.memory_opp[idx] 

        fig = plt.figure(figsize=(12, 8))
        ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=1)
        ax2 = plt.subplot2grid((3, 4), (0, 1), colspan=1)
        ax3 = plt.subplot2grid((3, 4), (0, 2), colspan=2)
        ax4 = plt.subplot2grid((3, 4), (1, 0), colspan=2)
        ax5 = plt.subplot2grid((3, 4), (1, 2), colspan=2)
        ax6 = plt.subplot2grid((3, 4), (2, 0), colspan=2)
        ax7 = plt.subplot2grid((3, 4), (2, 2), colspan=2)

        graph = [ [ax1,ax2,    ax3],
                  [  ax4,      ax5],
                  [  ax6,      ax7]]


        # sub_plots = plt.subplots(3, 2, figsize=(12, 8))
        # fig = sub_plots[0]
        # graph = sub_plots[1]
        # ax.set_xticks([0, 50, 100])
        # ax.set_yticks([-2, 0, 2])
        # ax.set_xticklabels(['start', 'middel', 'end'], fontsize=12)
        # ax.set_yticklabels(['low', 'zero', 'high'], fontsize=12)

        graph[0][0].hist(list_z)
        graph[0][0].set_title('z')
        graph[0][0].grid(visible=True, linestyle='--')
        #graph[0][0].set_xlabel('x')
        #graph[0][0].set_ylabel('y')

        graph[0][1].hist(list_color)
        graph[0][1].set_title('Board Colour')

        graph[0][2].hist(list_score, bins = 72)
        graph[0][2].set_title('score')
        graph[0][2].grid(visible=True, linestyle='--')

        graph[1][0].hist((list_B_a, list_B_b), bins = 10, color=['red', 'blue'])
        graph[1][0].set_title('Blocks B_a, B_b')
        graph[1][0].grid(visible=True, linestyle='--')
        graph[1][0].legend(['B_a','B_b']);

        graph[1][1].hist((list_D_a, list_D_b), bins = 36, color=['red', 'blue'])
        graph[1][1].set_title('Distance D_a, D_b')
        graph[1][1].grid(visible=True, linestyle='--')
        graph[1][1].legend(['D_a','D_b']);

        graph[2][0].hist(list_cnt, bins = 150)
        graph[2][0].set_title('Board Cnt')
        graph[2][0].grid(visible=True, linestyle='--')

        Narr_graph = np.zeros(140)
        for Narr in list_Narr :
            Narr_graph += Narr
        Narr_graph /= len(list_Narr)
        #print(Narr_graph)

        graph[2][1].plot(range(len(Narr_graph)), Narr_graph)
        graph[2][1].set_title('Narr elementwise average')
        graph[2][1].grid(visible=True, linestyle='--')

        fig.suptitle('Data Histogram')
        fig.tight_layout(pad=2)
        nowLog = datetime.datetime.now(timezone('Asia/Seoul'))     #<class 'datetime.datetime'> 2022-12-09 21:37:54.213080+09:00
        now_str = f"{nowLog.year}-{int(nowLog.month):02d}-{int(nowLog.day):02d}_{int(nowLog.hour):02d}h{int(nowLog.minute):02d}m{int(nowLog.second):02d}s" 

        saveas = saveas +"_" + now_str + ".png"
        fig.savefig(saveas)
        print(f"complete.  Saved as {saveas}")




class Trainer:

    def __init__(self, data_root = None, model_root = None):
      self.dualnet = None
      self.model_ver = None
      self.optimizer = None

      print("=========================================================================")
      print(f"trainer >> __init__ : Qzero Training         batchsize = {HPARAMS['BATCH_SIZE']}      DEVICE : {DEVICE}")
      print("=========================================================================")

    def train(self, data_root = "./data", model_root = "./models"):
      # ./model 폴더로부터 최신 모델을 가져와서, 데이터 최근 25만개로 학습
      self.dualnet, _, _, _ = self.load_checkpoint(folder = model_root)
      self.dualnet.to(DEVICE)
      self.dualnet.train()
      self.optimizer = torch.optim.SGD(self.dualnet.parameters(), \
                                                  lr=HPARAMS['LR'], momentum=HPARAMS['MOMENTUM_DECAY'], weight_decay=HPARAMS['C_L2'])
      print("Trainer >> Loading the Board_Dataset...")
      self.Train_dataset = Board_Dataset(data_root, hparameters = HPARAMS)
      self.Train_loader = torch.utils.data.DataLoader(self.Train_dataset,
                                        batch_size=HPARAMS['BATCH_SIZE'], 
                                        shuffle=True,
                                        drop_last=True)

      # TPARAMS['optimizer'] = self.optimizer
      # TPARAMS['train_dataset'] = self.Train_dataset
      # TPARAMS['train_loader'] = self.Train_loader
      
      batch_size = HPARAMS['BATCH_SIZE']
      C_g = HPARAMS['C_g']
      W_opp = HPARAMS['W_OPP']
      W_spdf = HPARAMS['W_SPDF']
      W_scdf = HPARAMS['W_SCDF']
      W_sbreg = HPARAMS['W_SBREG'] 
      W_scale = HPARAMS['W_SCALE']

      n_sudoepoches = min(ceil(250000/len(self.Train_dataset)) , 10 )
      n_epoches = self.Train_dataset.ReqEpoches
      torch.autograd.set_detect_anomaly(True)
      print("")
      print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      print(f"Trainer >> Starting Train.  Total SubEpoches : {n_epoches}    Sudo Epoches : {n_sudoepoches} (= min[ 10, max(250000/len(dataset), 1)])")
      print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      training_start_time = time.time()
      batch_cnt = 0
      loss_sum = 0
      tot_result = {}
      tot_result['Zloss'] = []
      tot_result['Ploss'] = []
      tot_result['OPPloss'] =[]
      tot_result['SBPloss'] = []
      tot_result['SBCloss'] = []
      tot_result['HUMloss'] = []
      tot_result['HUSloss'] = []
      tot_result['Penalty_Score'] = []
      for epoch in range(n_epoches):
          print(f"Processing sub-epoch {epoch+1}/{n_epoches}.. ")
          result = {}
          result['Zloss'] = []
          result['Ploss'] = []
          result['OPPloss'] =[]
          result['SBPloss'] = []
          result['SBCloss'] = []
          result['HUMloss'] = []
          result['HUSloss'] = []
          result['Penalty_Score'] = []
          for sudo_epoc in range(n_sudoepoches):
              for i_batch, batch in enumerate(tqdm(self.Train_loader)): 
                  self.optimizer.zero_grad()
                  
                  s =   batch['s'].to(DEVICE)
                  p =   batch['p'].to(DEVICE)       # Narr distribution of MCTS
                  opp = batch['opp'].to(DEVICE)     # action num of opp
                  z =   batch['z'].unsqueeze(-1).to(DEVICE)       # one-hot encoding of win/lose          
                  score = batch['score'].to(DEVICE) # one-hot encoding of the final score difference

                  ps_, vs_ = self.dualnet(s)
                  p_, opp_  = ps_[0], ps_[1]        # torch.Size([bs, 140])
                  v_, score_ = vs_[0], vs_[1]       # torch.Size([bs, 1]), torch_Size([bs, 41])
                  v_ = 0.5 * v_  + 0.5              # v_ = nn.Sigmoid()(v_)     

                  # Game outcome value loss, Policy loss: with focal loss
                  Zloss = C_g * -((torch.sum(torch.log(v_)*z)+ torch.sum(torch.log(1-v_)*(1-z)))  /batch_size)  
                  Ploss = -torch.sum(torch.log(p_)*p * ((1-p_)**2)  )/batch_size  


                  # Auxiliary Policy Targets loss (Opponent policy loss) :
                  OPPloss = W_opp * (-torch.sum(torch.log(opp_)*opp)/batch_size) 


                  # Score belief loss (“pdf”):
                  SBPloss = W_spdf * (-torch.sum(torch.log(score_)*score)/batch_size) 


                  # Score belief loss (“cdf”):
                  SBCloss = 0
                  for CDF_x in range(score.shape[1]):  # P1.shape = 5,  CDF_x = 0,1,2,3,4  마지막거는 어차피 0
                      SBCloss += torch.pow(torch.sum(score[:,:CDF_x+1].clone()- score_[:,:CDF_x+1],dim= 1,keepdim=True), 2.0)
                  SBCloss = W_scdf * (torch.sum(SBCloss)/batch_size)


                  # Score belief mean self-prediction:
                  HUMloss = 0
                  M_p = 0
                  M_phat = 0
                  for PDF_x in range(score.shape[1]):  #0~40
                      M_p += score[:,PDF_x].clone() * (PDF_x - 20)  # -20~20
                      M_phat += score_[:,PDF_x] * (PDF_x - 20)
                  HUMloss = nn.HuberLoss(reduction='mean', delta=10.0)(M_p.clone(), M_phat)
                  HUMloss = W_sbreg * HUMloss 

        
                  # Score belief standard deviation self-prediction:
                  HUSloss = 0
                  SIGMA_p = 0
                  SIGMA_phat = 0
                  for PDF_x in range(score.shape[1]): 
                      SIGMA_p += score[:,PDF_x].clone() * torch.pow((PDF_x - 20  - M_p.clone()), 2.0)  # always zero
                      SIGMA_phat += score_[:,PDF_x] * torch.pow((PDF_x - 20  - M_phat), 2.0)
                  SIGMA_phat = torch.sqrt(SIGMA_phat)
                  HUSloss = nn.HuberLoss(reduction='mean', delta=10.0)(SIGMA_p.clone(), SIGMA_phat) 
                  HUSloss = W_sbreg * HUSloss 

                  # Score belief scaling penalty:
                  V_head_scaling_params = []
                  Params = self.dualnet.V_head.scaling_component.parameters()
                  for param in Params:
                      V_head_scaling_params.append(param.view(-1))
                  Params = torch.cat(V_head_scaling_params)
                  Penalty_Score = W_scale * (torch.sum(torch.pow(Params,2.0)))

                  #print(f"Zloss{Zloss} Ploss{Ploss} OPPloss{OPPloss} SBPloss{SBPloss} SBCloss{SBCloss} HUMloss{HUMloss} HUSloss{HUSloss} Penalty_Score{Penalty_Score}")
                  loss = Zloss + Ploss + OPPloss + SBPloss + SBCloss + HUMloss + HUSloss + Penalty_Score
                  loss.backward()
                  loss_sum += loss.item()
                  result['Zloss'].append(Zloss.item())
                  result['Ploss'].append(Ploss.item())
                  result['OPPloss'].append(OPPloss.item()) 
                  result['SBPloss'].append(SBPloss.item())
                  result['SBCloss'].append(SBCloss.item())
                  result['HUMloss'].append(HUMloss.item()) 
                  result['HUSloss'].append(HUSloss.item())
                  result['Penalty_Score'].append(Penalty_Score.item())   
                  batch_cnt +=1     
                  self.optimizer.step()
              pass

          # Subepoch End. plot losses    
          print(f"subepoch {epoch+1} finished, Zloss: {round(Zloss.item(),3)} Ploss: {round(Ploss.item(),3)} OPPloss: {round(OPPloss.item(),3)} SBPloss: {round(SBPloss.item(),3)} SBCloss: {round(SBCloss.item(),3)} HUMloss: {round(HUMloss.item(),3)} HUSloss: {round(HUSloss.item(),3)} Penalty_Score: {round(Penalty_Score.item(),3)}\n\n")
          self.plot_losses(result, epoch+1)
          for key, val in result.items():
              tot_result[key] += result[key]
          
          # If need more Subepoches, load data to memory
          if (epoch+1) != n_epoches :
              self.Train_dataset.LoadData_ToMemory() 
              self.Train_loader = torch.utils.data.DataLoader(self.Train_dataset,
                                                batch_size=HPARAMS['BATCH_SIZE'], 
                                                shuffle=True,
                                                drop_last=True)
          pass

      print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      print(f"Trainer >> Ended train.   elapsed time :{get_ElapsedStr(training_start_time)}")
      print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      self.dualnet.eval()

      # Plot all Subepoches losses, and all history logs (until now)
      self.plot_losses(tot_result, 777, entire = True)
      self.plot_losses_history(folder = "./models")

      # return all Subepoches losses log
      tot_result['loss_sum'] = loss_sum/  (len(self.Train_loader) * n_sudoepoches * n_epoches)
      return tot_result

    def plot_losses_history(self, folder = "./models") :

        models_log_list = sorted(glob.glob(f'{folder}/*.pth'))      # ['./models/dual_net[6,96]-0001.pth', './models/dual_net[6,96]-0002.pth',]
        models_log_list = models_log_list[2:]
        history = {}
        history['loss_sum'] = []
        history['Zloss'] = []
        history['Ploss'] = []
        history['OPPloss'] =[]
        history['SBPloss'] = []
        history['SBCloss'] = []
        history['HUMloss'] = []
        history['HUSloss'] = []
        history['Penalty_Score'] = []

        for log_model in models_log_list :
            info = torch.load(log_model, map_location=torch.device(DEVICE))
            result = info['Training Loss log']
            for key, val in result.items():
                if key == "loss_sum" :
                    history[key].append(result[key])
                else :
                    history[key] += result[key]

        # Create a figure for loss_sum logs
        fig, axs = plt.subplots(figsize=(30, 10))
        axs.plot(history['loss_sum'], '-b', label='TotalLoss')
        axs.set_title("Loss Sum history")
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True)
        fig.suptitle("Qzero Neural Net Training Losses - Total_History")
        nowLog, forms =  get_nowtime()    
        now_str = forms[2]
        saveas = f"./log/Trainer/TotalLossHistory" + ".png"
        fig.savefig(saveas)
        print(f"Training History (Total lossess) saved as {saveas}")

        # Create a figure for all other lossess logs
        self.plot_losses(history, 777, history = True)
        pass


    def save_checkpoint(self, result = None, folder = "./models", filename = None) :
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """

        nowLog, forms =  get_nowtime()    
        now_str = forms[2]

        #recent_package_idx = self.check_data()
        B, C, ver =  re.findall(r'\d+',self.model_path )    
        name = filename
        if filename is None :
            name = f"dual_net[{B},{C}]-{(int(ver)+1):05d}.pth"

        if HPARAMS['SWA'] :
            old_model = getModel(int(B),int(C))
            old_model.load_state_dict( torch.load(self.model_path, map_location=torch.device(DEVICE))['model_state_dict'] )            
            
            sdA = old_model.state_dict()
            sdB = self.dualnet.state_dict()
            # Average all parameters
            decay = HPARAMS['SWA_decay']
            for key in sdA:
                sdB[key] = ( (1- decay) * sdB[key] + decay * sdA[key]  )

            # Load averaged state_dict (or use modelA/B)
            self.dualnet.load_state_dict(sdB) 

        torch.save({'model_state_dict': self.dualnet.state_dict(),
                    'optimizer_state_dict':self.optimizer.state_dict(),
                    #'recent_package_idx': recent_package_idx,
                    'Training Loss log' : result,
                    'HPARAMS' : HPARAMS,
                    'comment' : f"saved at {now_str}",  
                    }, f"{folder}/{name}")

        print(f"{folder}/{name}")
        self.model_ver = int(ver) + 1
        return self.model_ver



    def load_checkpoint(self, folder = "./models", filename = None) :
        """
        return max(glob(./models))
        """
        recent_model = max(glob.glob(f'{folder}/*.pth'))       # ['./models/dual_net[6,96]-0001.pth', './models/dual_net[6,96]-0002.pth',]
        self.model_path = recent_model 
        B, C, ver =  re.findall(r'\d+',recent_model)         

        print(f"Trainer >> Loading the most recent model in {folder}.. dual_net[b{B}c{C}]-{ver}")
        model = getModel(int(B),int(C))
        model.load_state_dict( torch.load(recent_model, map_location=torch.device(DEVICE))['model_state_dict'] )   
        self.model_ver = int(ver)
        return model, B, C, ver
   


    def load_champion(self, folder = "./models/Champion", filename = None) :
        return self.load_checkpoint(folder)



    def update(self, d_root = "./data", m_root = "./models"):

        #### loggingFlag ##### 
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"Updater >> Training starts... \tcalling self.train()")
        result = self.train(data_root = d_root, model_root = m_root)
        print(f"Updater >> Training complete. ")

        print("Updater >> Saving checkpoint... \tcalling self.save_checkpoint()")
        new_model_ver = self.save_checkpoint(result, folder = m_root, filename = None)
        print(f"Updater >> Saving checkpoint complete.  Model Version updated :{new_model_ver-1} -> {new_model_ver}")

        if HPARAMS['HitCntUpdate'] :
            print("Updater >> Revising data hit count... \tcalling self.dataset.hit_cnt_update()")
            self.Train_dataset.hit_cnt_update(folder = d_root)
            print("Updater >> Revising data hit count complete. sleep for 60 secs...")
            time.sleep(60)  # wait for google drive updating data hit counts properly

        print(f"Updater >> Training result = \n{result}")
        print(f"Updater >> End Trainer. -> new trained model Version {self.model_ver} ")

        
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        #######################
        
    def check_zeroHit_samplesNum(self, folder = "./data"):
        # ./data 폴더에 있는 hit count == 0 인 총 samples 수 게산
        data_list = glob.glob(f'{folder}/*.pkl')              # ['./data/package[2004,b5c64,6000,2,0]-[2022-12-28_22h-28m-11s-142152].pkl', ...]
        if data_list :
            num_samples_sum = 0
            for idx in range(len(data_list)):
                samples_num, _, _, _, _, currHit, _, _, _, _, _, _, _,  =  re.findall(r'\d+',data_list[idx])
                if int(currHit) == 0 :
                    num_samples_sum += int(samples_num)
            return num_samples_sum
        return 0         
    
    def Mutex_train_lock(self, folder = "./log") :
        text_file_path = f"{folder}/Mutex_Train.txt"

        nowLog, forms =  get_nowtime()    
        now_str = forms[2]
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
                print(f"Trainer >> Lock Mutex_Train Success, {now_str}")
            elif '1' in lockLog :
                nowLog_rec = datetime.datetime.strptime(now_str, '%Y-%m-%d %H:%M:%S')
                pastLog_rec = datetime.datetime.strptime(pastLog, '%Y-%m-%d %H:%M:%S')     
                diff = nowLog_rec - pastLog_rec
                print(f"Trainer >> Achieve Mutex_Train_lock Failed. Someone is already training model for {diff}")
                if HPARAMS['TRAIN_TIMEOUT'] < diff :
                    lock_achieved = True
                    edited_lines.append("1\n")
                    print(f"Trainer >> Train_lock TIMEOUT :: abort lock. Someone maybe dead. He was training for {diff}")
        if lock_achieved == True :
            with open(text_file_path,'w') as f:
                f.writelines(edited_lines)
        return lock_achieved

    def Mutex_train_unlock(self,folder = "./log") :
        text_file_path = f"{folder}/Mutex_Train.txt"

        nowLog, forms =  get_nowtime()    
        now_str = forms[2]

        edited_lines = [f'{now_str}\n','0\n']
        with open(text_file_path,'w') as f:
            f.writelines(edited_lines)
        print(f"Trainer >> Unlock Mutex_Train Success, {now_str}")


    def step(self, data_root = "./data", model_root = "./models"):
      if self.Mutex_train_lock() :
          total_samples_num = check_totalNum_samples(folder = data_root)
          new_data_num = self.check_zeroHit_samplesNum(folder = data_root)
          print(f"Trainer >> Current total {total_samples_num}samples / new samples : {new_data_num} (0 hit) ")
          if new_data_num >= 250000 : 
              print(f"Trainer >> Sufficient Data colleted that are not trained : {new_data_num} >= 250000 ")
              self.update()
          else :
              print(f"Trainer >> not enough data. next model update is when {new_data_num}(new samples) >= 250000 ")
              print(f"Trainer >> If you want to train anyway, Use  trainer.update()  instead of  trainer.step()")
          self.Mutex_train_unlock() 
      pass

    def plot_losses(self,losses, epoch, entire = False, history = False):
        # Extract the losses for each of the eight loss functions
        loss_1, loss_2, loss_3, loss_4 = losses['Zloss'], losses['Ploss'], losses['OPPloss'], losses['SBPloss']
        loss_5, loss_6, loss_7, loss_8 = losses['SBCloss'], losses['HUMloss'], losses['HUSloss'], losses['Penalty_Score']
       
        # Create a figure with eight subplots
        fig, axs = plt.subplots(2, 4, figsize=(30, 10))
        
        # Plot the losses for each of the eight loss functions
        axs[0, 0].plot(loss_1, '-b', label='Loss 1')
        axs[0, 0].set_title("Loss 1 : Zloss")
        axs[0, 1].plot(loss_2, '-g', label='Loss 2')
        axs[0, 1].set_title("Loss 2 : Ploss")
        axs[0, 2].plot(loss_3, '-r', label='Loss 3')
        axs[0, 2].set_title("Loss 3 : OPPloss")
        axs[0, 3].plot(loss_4, '-c', label='Loss 4')
        axs[0, 3].set_title("Loss 4 : SBPloss")
        axs[1, 0].plot(loss_5, '-m', label='Loss 5')
        axs[1, 0].set_title("Loss 5 : SBCloss")
        axs[1, 1].plot(loss_6, '-y', label='Loss 6')
        axs[1, 1].set_title("Loss 6 : HUMloss")
        axs[1, 2].plot(loss_7, '-k', label='Loss 7')
        axs[1, 2].set_title("Loss 7 : HUSloss")
        axs[1, 3].plot(loss_8, '-k', label='Loss 8')
        axs[1, 3].set_title("Loss 8 : Penalty_Score")
        
        # Add a legend
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True)
        
        if entire :
            epoch = "All"
        if history :
            epoch = "Total_History"
        # Add a title
        fig.suptitle("Qzero Neural Net Training Losses - Epoch {}".format(epoch))


        nowLog, forms =  get_nowtime()    
        now_str = forms[2]

        if not history :
            B, C, ver =  re.findall(r'\d+',self.model_path)   
            model_info = f"dual_net[{B},{C}]-{ver}"      # dual_net[5,64]-00003.pth

            output_path = f"./log/Trainer/Log_{model_info}"
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            saveas = f"{output_path}/epoch-{str(epoch)}_{now_str}" + ".png"
        if history :
            saveas = f"./log/Trainer/{epoch}" + ".png"
        fig.savefig(saveas)
        print(f"Training log saved as {saveas}")

        plt.close('all') 
        pass

def Init_HitCnt_toZero(folder = "./data"):
    """
    initialize hit cnt of all package to 0.
    """
    for file_path in glob.glob(folder + '/*.pkl'):
        old_name = file_path              # ['./data/package[2004,b5c64,6000,2,0]-[2022-12-28_22h-28m-11s-142152].pkl', ...]
        di = re.findall(r'\d+',old_name)

        pack_info = f"[{di[0]},b{di[1]}c{di[2]},{di[3]},{di[4]},0]"
        date_info = f"[{di[6]}-{di[7]}-{di[8]}_{di[9]}h-{di[10]}m-{di[11]}s-{di[12]}]"

        new_name = f'{folder}/package{pack_info}-{date_info}.pkl'
        os.rename(old_name, new_name)        
        time.sleep(0.1)
    pass

if __name__ == "__main__":
    import os
    print("Running CyTrainer.py in cwd :",os.getcwd(), end="\n\n")
    # hit cnt 0인 데이터가 25만개이상이면 Mutex_Train_Lock 걸고 훈련 시작  
    
    Qzero_Trainer = Trainer()
    Qzero_Trainer.plot_losses_history()

    Qzero_Trainer.Mutex_train_unlock() 
    for i in range(20) :
        Qzero_Trainer.step(data_root = "./data", model_root = "./models")  

    # Qzero_Trainer.update(d_root = "./data", m_root = "./models")  # 데이터 개수 상관 없이 최근Nwindow 강제 업데이트
    #Init_HitCnt_toZero()           # 데이터의 hit cnt 모두 0으로 초기화

