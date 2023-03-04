'''
YAI-quoridor/src/Settings.py
'''
from pytz import timezone
import datetime
import time
import torch
import sys
import glob
import pickle
import re
import os
from math import ceil
from math import floor

sys.path.append("/content/cpp")
sys.path.append("/content/drive/MyDrive/Colab Notebooks/YAI-quoridor")
sys.path.append("/content/drive/MyDrive/Colab Notebooks/YAI-quoridor/src")


NOTES = 'Qzero'
START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


## (100,600), handeval 일때 0.3sec per game
## (1000,6000), handeval 일때 2.7sec per game
HPARAMS = {
    # use NN    
    'useNN' : False,                   # If False, will not use inference. use heuristics
    'bandit': None,                    # if None, AUTO selected below

    #train when selfplay
    'selfplay_with_train' : False,

    #arena when selfplay
    'selfplay_with_arena' : False,

    #Data save?
    'selfplay_with_DataSave' : False,

    #Stat save?
    'selfplay_with_Verbose' : False,
    'arena_with_Verbose' : True,

    #player
    'PLAYOUT_N_FULL' : 6000,
    'PLAYOUT_n_SMALL' : 1000,          #disable all explorative settings, (e.g. dirichlet noice)maximizing strength 
    'INFERENCE_BATCH_NUM' : 1024,      # player num
    'printInterval' : None,                # if None, AUTO selected (Proper Frequently) below
    'data_buffer_sz' : 2000,            #how many samples of a single data package 

    #Arena
    'Arena_PLAYOUT_N' : 10000,
    'Arena_GAME_NUM': 100,
    'CHANGE_CHAMPION': False,
    'AUTOSELECT' : False,
    'SaveStatisticsProbability': 1,

    #trainer
    'BATCH_SIZE': 1024,
    'HitCntUpdate' : False,
    'LoadMemoryPartition' : 300000,    # 1 epoch has how many samples
    'SWA' : False,
    'SWA_decay' : 0.75,
    #Data Augmentation
    'use_Augmentation' : True,
    'Augment_randomly' : True,    # If True, Augmentation will not increase total training data. if False, then data will increase to X4
      #optimizer
      'MOMENTUM_DECAY': 0.9,
      'LR': 6e-5,
      'C_L2'  : 3e-5,       
      #loss
      'C_g'   : 1.5,
      'W_OPP' : 0.15,
      'W_SPDF' :0.02,
      'W_SCDF' :0.02,
      'W_SBREG':0.004,
      'W_SCALE' : 0.0005,

    #gating
    'GATING_snapshots_no' : 4,
  
    #Mutex lock timeout
    'TRAIN_TIMEOUT' : datetime.timedelta(minutes = 20, seconds = 30),
    'ARENA_TIMEOUT' : datetime.timedelta(hours = 1),
}
HPARAMS['bandit'] = ("PUCT" if HPARAMS['useNN'] is True else "impUCB" ) if HPARAMS['bandit']  is None else HPARAMS['bandit']
if not HPARAMS['useNN'] :
    HPARAMS['INFERENCE_BATCH_NUM'] = 1            # cpu만 쓸때 0.3sec/game(batch1024) -> 0.16sec/game(batch1).  MCTS 전체가 cache에 올라갈 수 있음 
HPARAMS['printInterval'] = ((ceil(HPARAMS['INFERENCE_BATCH_NUM']/10) + 1) if HPARAMS['useNN'] else 1000) if HPARAMS['printInterval'] is None else HPARAMS['printInterval']

PLAYOUT_FULL = HPARAMS['PLAYOUT_N_FULL']
PLAYOUT = HPARAMS['PLAYOUT_n_SMALL']

TPARAMS = {
    'training_date' : START_DATE,
    'model_ver' : None,
    'NNname' : None,
    'B' : None,
    'C' : None,    
    'dual_net' : None,
    'models' : None,

    # 'optimizer' : None,
    # 'train_dataset' : None,
    # 'train_loader' : None,
    'trainer' : None,

    'games_count' : None,
    'player_no' : 0 ,
    'Arena_player_no' : 0,
}

##########################################################################
######################### utils function #################################
##########################################################################
# package 저장 포맷 변경 반영 : settings 내 함수들, dataset.hit_cnt_update, Cytrainer.hit_cnt_allZero
# model 저장 포맷 변경 반영 : 
##########################################################################

def get_ElapsedStr(starttime):
    return f"{floor((time.time() - starttime)/60)} min {round((time.time() - starttime) % 60,3)} secs"

def get_nowtime():
    nowLog = datetime.datetime.now(timezone('Asia/Seoul'))     #<class 'datetime.datetime'> 2022-12-09 21:37:54.213080+09:00
    form1 = f"{nowLog.year}-{nowLog.month}-{nowLog.day}_{nowLog.hour}h-{nowLog.minute}m-{nowLog.second}s"  #"2022-12-9_21h-37m-54s"
    form2 = f"{nowLog.year}-{nowLog.month}-{nowLog.day} {nowLog.hour}:{nowLog.minute}:{nowLog.second}"  #"2022-12-9 21:37:54" , now.microsecond = 087337
    form3 = f"{nowLog.year}-{int(nowLog.month):02d}-{int(nowLog.day):02d}_{int(nowLog.hour):02d}h{int(nowLog.minute):02d}m{int(nowLog.second):02d}s"  # "2022-12-09_21h37m05s"
    forms = [form1, form2, form3]
    return nowLog, forms
    
def save_stat(stat):
    data_list = glob.glob('./log/Statistics/*.pkl')                           
    recent_package_idx = 0
    if any(data_list) :
        recent_package_idx = re.findall(r'\d+', max(data_list))[0]
    new_idx = int(recent_package_idx) + 1
    stat['data_idx'] = new_idx

    filename = f'./log/Statistics/Stats-{new_idx:05d}.pkl'
    with open(filename,'wb') as f:
        pickle.dump(stat, f, pickle.HIGHEST_PROTOCOL)    
    print(f"Saving Stats.. {filename}")    
    pass

def check_totalNum_samples(folder = "./data"):  # CyTrainer.step, BoardDataset.init
    """
    ./data 폴더에 있는 총 samples 수 계산
    """
    data_list = glob.glob(f'{folder}/*.pkl')                 # ['./data/package-00001[1500,b6c96,0].pkl', './data/package-00002[1500,b6c96,0].pkl',]
    if data_list :
        num_samples_sum = 0
        for idx in range(len(data_list)):
            samples_num, _, _, _, _, _, _, _, _, _, _, _, _,  =  re.findall(r'\d+',data_list[idx])
            num_samples_sum += int(samples_num)
        return num_samples_sum
    return 0 

def get_Nwindow_RequiredHit(folder = "./data"):
    alpha = 0.75
    beta = 0.4
    cc = 250000
    Ntotal = check_totalNum_samples(folder) 

    recent_model = max(glob.glob(f'./models/*.pth'))       # ['./models/dual_net[6,96]-0001.pth', './models/dual_net[6,96]-0002.pth',]
    _, _, ver =  re.findall(r'\d+',recent_model)         

    #Required_hit = (1 + beta*( (  ((Ntotal/cc)**(alpha)) -1)  /alpha )  )
    Required_hit = (1 + beta*( (  ((int(ver))**(alpha)) -1)  /alpha )  )
    Nwindow = round(cc * Required_hit)
    Required_hit = ceil(Required_hit)

    #print(f"Train Dataset >> Checked Nwindow  = {Nwindow}, based on N_samples = {Ntotal} ...complete.")
    return Nwindow, Required_hit 

def save_data(data,B,C,ver):
    """
    최신 데이터 패키지 이름 조회, 인덱스 확인 후  
    데이터를 package[샘플수,모델구조,Playout수,필요hit수,현재hit수]-[2022-12-28__22h-28m-11s_142152]  이름으로 저장
    package[2004,b5c64,6000,2,0]-[2022-12-28_22h-28m-11s-142152]
    """
    nowLog = datetime.datetime.now(timezone('Asia/Seoul'))     #<class 'datetime.datetime'> 2022-12-09 21:37:54.213080+09:00
    now_str = f"{nowLog.year}-{int(nowLog.month):02d}-{int(nowLog.day):02d}_{int(nowLog.hour):02d}h-{int(nowLog.minute):02d}m-{int(nowLog.second):02d}s-{nowLog.microsecond}"  #2022-12-9_21h-37m-54s_213080" , now.microsecond = 087337

    if not any(data) :
        print("Fatal Error, databuffer is empty list !")
        exit(0)
    samples_num = len(data)

    model_type = "b0c0"
    if HPARAMS['useNN'] :
        model_type = f"b{B}c{C}"
    
    pp = HPARAMS['PLAYOUT_N_FULL']

    _, Required_hit = get_Nwindow_RequiredHit()

    filename = f'./data/package[{samples_num},{model_type},{pp},{Required_hit},0]-[{now_str}].pkl'    

    if HPARAMS['selfplay_with_DataSave'] :
        with open(filename,'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

# def modify_dataName(folder = "./data"):
#     data_list = glob.glob(f'{folder}/*.pkl')                 # ['./data/package-00001[1500,b6c96,0].pkl', './data/package-00002[1500,b6c96,0](1).pkl',]
#     nowLog = datetime.datetime.now(timezone('Asia/Seoul')) 
#     now_str = f"{nowLog.year}-{nowLog.month}-{nowLog.day}_{nowLog.hour}h-{nowLog.minute}m-{nowLog.second}s-{nowLog.microsecond}"
#     _, Req_hit = get_Nwindow_RequiredHit()
#     if data_list :
#         for idx in range(len(data_list)):
#             int_list  =  re.findall(r'\d+',data_list[idx])
#             pidx, Nsam, block, chan, currhit = int(int_list[0]), int(int_list[1]), int(int_list[2]), int(int_list[3]), int(int_list[4])
#             os.rename(data_list[idx], f'./data/package[{Nsam},heuristic-b0c0,{6000},{Req_hit},0]-[].pkl')
#     print("modify data name complete.")

