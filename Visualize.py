import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob

class DisplayBoard:

    def __init__(self):
        #redo-able
        self.blocks_vertical = [[0 for i in range(8)] for i in range(8)]
        self.blocks_horizon = [[0 for i in range(8)] for i in range(8)]
        self.a = [8,4]
        self.b = [0,4]
        self.a_remain = 10
        self.b_remain = 10
        self.count = 0
        self.colour = 0
        self.history = []

    def doAction(self, num):

        if num < 64:
            x = num // 8
            y = num % 8
            self.blocks_vertical[x][y] = 1
            if self.colour == 0:
                self.a_remain -= 1
            else:
                self.b_remain -= 1
        elif num < 128:
            x = (num-64) // 8
            y = (num-64) % 8
            self.blocks_horizon[x][y] = 1
            if self.colour == 0:
                self.a_remain -= 1
            else:
                self.b_remain -= 1
        else:
            player = self.a if self.colour == 0 else self.b

            if num == 128:
                player[0] -= 1
            elif num == 129:
                player[1] += 1
            elif num == 130:
                player[1] -= 1
            elif num == 131:
                player[0] += 1
            elif num == 132:
                player[0] -= 2
            elif num == 133:
                player[1] += 2
            elif num == 134:
                player[1] -= 2
            elif num == 135:
                player[0] += 2
            elif num == 136:
                player[0] -= 1
                player[1] += 1
            elif num == 137:
                player[0] -= 1
                player[1] -= 1
            elif num == 138:
                player[0] += 1
                player[1] += 1
            elif num == 139:
                player[0] += 1
                player[1] -= 1

        self.colour = 1 - self.colour
        self.count += 1

        self.history.append(num)

    def undo(self):
        self.winner = 0
        self.colour = 1 - self.colour
        self.count -= 1
        
        num = self.history.pop()

        if num < 64:
            x = num // 8
            y = num % 8
            self.blocks_vertical[x][y] = 0
            if self.colour == 0:
                self.a_remain += 1
            else:
                self.b_remain += 1
            return num
        elif num < 128:
            x = (num-64) // 8
            y = (num-64) % 8
            self.blocks_horizon[x][y] = 0
            if self.colour == 0:
                self.a_remain += 1
            else:
                self.b_remain += 1
            return num

        player = self.a if self.colour == 0 else self.b

        if num == 128:
            player[0] += 1
        elif num == 129:
            player[1] -= 1
        elif num == 130:
            player[1] += 1
        elif num == 131:
            player[0] -= 1
        elif num == 132:
            player[0] += 2
        elif num == 133:
            player[1] -= 2
        elif num == 134:
            player[1] += 2
        elif num == 135:
            player[0] -= 2
        elif num == 136:
            player[0] += 1
            player[1] -= 1
        elif num == 137:
            player[0] += 1
            player[1] += 1
        elif num == 138:
            player[0] -= 1
            player[1] -= 1
        elif num == 139:
            player[0] -= 1
            player[1] += 1

        return num



def draw_walls(ax, Vwalls, Hwalls):
    """
    draw quoridor walls on a matplotlib axis
        % input
          ax : the matplotlib axis
          Vwalls,Hwalls : np.array(board.blocks_vertical)    e.g.  Vwalls.shape =(8,8) <numpy ndarray>
    """  
    for i in range(8) :
        for j in range(8) :
            y,x = i,j
            y = 8-y
            if Vwalls[i,j] == 1 :
                rect = matplotlib.patches.Rectangle((x+1-0.11, y-1),
                                                0.2, 2,
                                                color ='brown')
                ax.add_patch(rect)
            if Hwalls[i,j] == 1 :
                rect = matplotlib.patches.Rectangle((x, y-0.09),
                                                2, 0.2,
                                                color ='brown')
                ax.add_patch(rect)
    pass

def draw_path_edges(ax, hori_edges, vert_edges, colour, label):
    """
    draw quoridor walls on a matplotlib axis
        % input
          ax : the matplotlib axis
          hori_edges : np.array(8,9)   
          vert_edges : np.array(9,8)    
    """  
    if label == True :
        criterion = 1
    else :
        criterion = 0


    colour = "blue"

    # plot horizontal edges 0,0 ~ 7,8
    for i in range(8) :
        for j in range(9) :
            y,x = i,j
            #y = 9-y
            if hori_edges[i,j] >= criterion :
                rect = matplotlib.patches.Rectangle((x - 0.05, y ),
                                                0.1, 1,
                                                color =colour, alpha = hori_edges[i,j])  
                ax.add_patch(rect) 
                #ax.text(x + 0.5 - 0.05, y - 1.5, str(hori_edges[i,j]), ha='center', va='center') 
    for i in range(9) :
        for j in range(8) :
            y,x = i,j
            #y = 9-y          
            if vert_edges[i,j] >= criterion :
                rect = matplotlib.patches.Rectangle((x , y - 0.05),
                                                1, 0.1,
                                                color =colour, alpha = vert_edges[i,j])    
                ax.add_patch(rect)  
                #ax.text(x + 0.5 , y - 0.5 - 0.05, str(vert_edges[i,j]), ha='center', va='center')     
    pass

def draw_stone(ax,loc,color):
    """
    draw quoridor players on a matplotlib axis
        % input
          ax : the matplotlib axis
          loc : the location of player.   e.g. [8,4]  <python list>
          color : the color of player, 0 or 1   <int>
    """    
    y,x = loc
    x +=0.5 ;y += 0.5
    y = 9-y
    color = 'b' if color else 'r'
    stone = ax.plot(x,y,'o',markersize=28, 
                      markeredgecolor=(0,0,0), 
                      markerfacecolor=color, 
                      markeredgewidth=1)
    pass      

def draw_board(ax, stones, Vwalls, Hwalls, Stat):
    """
    draw specific quoridor board on a matplotlib axis
        % input
          ax : the matplotlib axis
          stones : [board.a,board.b]  e.g. [[8,4],[0,4]] <python list>
          Vwalls,Hwalls : np.array(board.blocks_vertical)    e.g.  Vwalls.shape =(8,8) <numpy ndarray>
          Stat : <python dict>
    """
    ax.cla()  
    # turn off the axes
    ax.set_axis_off()    
    # draw the vertical lines, horizontal lines
    for i in range(10):
        ax.plot([i, i], [0, 9], color='k')
        ax.plot([0, 9], [i, i], color='k')

    for x in range(9):
        for y in range(9):
            ax.text(x+0.5, y+0.5, f"{x},{8-y}", ha='center', va='center')    

    # Label the rows and columns
    for i in range(10):
        ax.text(i, -0.5, str(i), ha='center', va='top')
        ax.text(-0.5, i, chr(ord('A') + i), ha='right', va='center')   

    #ax.set_position([0.125, 0.38021, 0.33632, 0.49978])    
    #print(ax.get_position().bounds)     #(0.125, 0.38021126760563384, 0.33632075471698114, 0.49978873239436616) x,y,w,h
    for idx, stone in enumerate(stones) :
        draw_stone(ax,stone,idx)
    draw_walls(ax,Vwalls,Hwalls)   
    if Stat is not None :
        ax.title.set_text(f'P1 (r) : {Stat["NNname"][0]}\nP2 (b) : {Stat["NNname"][1]}')

def display_qzero(Stat= None, id = None):
    # create a figure to draw the game statistics
    """
      | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
      ----------------------------------------------------
    1 |             V             |
    2 |             Q             |
    3 |             D             |          보드
    4 |            pp             |
    5 |                           |_______________________
    6 |             
    7 |
    8 |
    9 |

    """

    fig = plt.figure(figsize=(18, 12))
    grid_ = (6*2,9)
    b_w, b_h = 4, 4
    ax1_board = plt.subplot2grid(grid_, (0, 0), colspan=b_w, rowspan=b_h*2)
    ax2 = plt.subplot2grid(grid_, (0, 4), colspan=5, rowspan = 2)   #not valid after r moves
    ax3 = plt.subplot2grid(grid_, (2, 4), colspan=5, rowspan = 2)
    ax4 = plt.subplot2grid(grid_, (4, 4), colspan=5, rowspan = 2)   
    ax5 = plt.subplot2grid(grid_, (6, 4), colspan=5, rowspan = 2)
    ax6 = plt.subplot2grid(grid_,(8+1,0), colspan=4, rowspan=3)
    ax7 = plt.subplot2grid(grid_,(8+1,4), colspan=5, rowspan=3)   #not valid after r moves

    axs = {'V_history' : ax2, 
           'Q_history' : ax3,
           'D_history' : ax4,
           'pp_history' : ax5,
            #'sum_R_list' : ax5,
           'B_history' : ax6,
           'Score_history' : ax7, }

        # # Pre-Record Statistics
        # self.Stat['V_history'] = {0: [], 1: [] }
        # self.Stat['rootchild'] = {0: [], 1: [] }    # (action, N, Q*10^7, V*10^7, P*10^7 ) #(int) (child->Q*10000000), (int) (child->V*10000000), (int) (child->P*10000000)
        # ==not use===== self.Stat['Score_history'] = {0: [], 1: [] }
        # ==not use===== self.Stat['Parr_history'] = {0: [], 1: [] }

        # # Post-Record Statistics
        # self.Stat['pp_history'] = {0: [], 1: [] }
        # ==not use===== self.Stat['Narr_history'] = {0: [], 1: [] }
        # self.Stat['Q_history'] = {0: [], 1: [] }
        # self.Stat['B_history'] = {0: [], 1: [] }
        # self.Stat['D_history'] = {0: [], 1: [] }
        # self.Stat['history'] = []

        # self.Stat['is_SelfPlay'] = True
        # self.Stat['NNname'] = {0: TPARAMS['NNname'], 1: TPARAMS['NNname'] }

    if Stat is not None :
        leng = len(Stat['history'])
        cnt = np.zeros(leng)
        for key, value in axs.items() :
            if key == 'Score_history'  :
                ScorePreds0, ScorePreds1 = calculate_score_mean(key)
                axs[key].plot(ScorePreds0, marker="o", label="p1", color= 'r')
                axs[key].plot(ScorePreds1, marker="o", label="p2", color= 'b') 
            else :
                axs[key].plot(Stat[key][0], marker="o", label="p1", color= 'r')
                axs[key].plot(Stat[key][1], marker="o", label="p2", color= 'b')
            if id is not None :
                axs[key].axvline(x = math.floor(id/2), color = 'k', linestyle='--', linewidth=3)
            axs[key].legend()
            axs[key].grid(True)


        fig.suptitle(f"Stats-{Stat['data_idx']:05d}", fontsize = 14)    
        ax2.title.set_text('V (NeuralNet inference Value head)')
        ax3.title.set_text('Q (After MCTS, playout == (n or N))')
        ax4.title.set_text('D (Shortest Path Distance)')
        ax5.title.set_text('PP (# of Playout)')
        ax6.title.set_text('B (blocks left)')
        ax7.title.set_text('Sco (Score predicted)')

    # manupulate the gap between subplots
    plt.tight_layout()
    # set the background color
    fig.patch.set_facecolor((0.85,0.64,0.125))  
    return fig, ax1_board

def calculate_score_mean(key):
    ScorePreds0 = []
    ScorePreds1 = []
    for Score_arr in Stat[key][0]:  # Score_arr.shape = (41,)
        M_p = 0
        for PDF_x in range(Score_arr.shape[0]):  #0~40
            M_p += Score_arr[PDF_x] * (PDF_x - 20)  # -20~20
        ScorePreds0.append(M_p)
    for Score_arr in Stat[key][1]:  # Score_arr.shape = (41,)
        M_p = 0
        for PDF_x in range(Score_arr.shape[0]):  #0~40
            M_p += Score_arr[PDF_x] * (PDF_x - 20)  # -20~20
        ScorePreds1.append(M_p)               
   
    return ScorePreds0, ScorePreds1

def load_recent_stats(num_stats=3) :
    
    # view most recent num_stats # of games
    data_list = glob.glob('./log/Statistics/*.pkl')    
    Statistics = []                       
    for idx, path in enumerate(reversed(data_list)) : #reversed(sorted(data_list)) :
        print(f"Loading {path}")
        with open(path, 'rb') as f : 
            Stat = pickle.load(f) 
            Statistics.append(Stat)
        if idx == num_stats-1 :    
            break
    for idx in range(num_stats) :
        Stat = Statistics[idx]
        board = DisplayBoard()  
        for idx,action in enumerate(Stat['history']):
            board.doAction(action)
            fig, ax1 = display_qzero(Stat, id = idx)
            draw_board(ax1, [board.a,board.b], np.array(board.blocks_vertical), np.array(board.blocks_horizon), Stat)
            plt.pause(0.05) #is necessary for the plot to update for some reason
            plt.show()
            print(board.history)

def load_specific_stat(data_idx = 1) :
    if data_idx == 0 :
        data_idx = 1
    data_list = glob.glob('./log/Statistics/*.pkl') 
    path = data_list[data_idx-1]
    Statistics = []  
    print(f"Loading {path}")
    with open(path, 'rb') as f : 
        Stat = pickle.load(f) 
    board = DisplayBoard()  
    for idx,action in enumerate(Stat['history']):
        board.doAction(action)
        fig, ax1 = display_qzero(Stat, id = idx)
        draw_board(ax1, [board.a,board.b], np.array(board.blocks_vertical), np.array(board.blocks_horizon), Stat)
        plt.pause(0.05) #is necessary for the plot to update for some reason
        plt.show()
        print(board.history)

def load_range_stat(start = 0, end = 3) :
    # view start ~ end   idx Stats
    data_list = glob.glob('./log/Statistics/*.pkl')    
    Statistics = []                       
    for idx, path in enumerate(data_list) : #reversed(sorted(data_list)) :
        if idx >= start-1 and idx <= end-1 :
            print(f"Loading {path}")
            with open(path, 'rb') as f : 
                Stat = pickle.load(f) 
                Statistics.append(Stat)
    for idx in range(len(Statistics)) :
        Stat = Statistics[idx]
        board = DisplayBoard()  
        for idx,action in enumerate(Stat['history']):
            board.doAction(action)
            fig, ax1 = display_qzero(Stat, id = idx)
            draw_board(ax1, [board.a,board.b], np.array(board.blocks_vertical), np.array(board.blocks_horizon), Stat)
            plt.pause(0.05) #is necessary for the plot to update for some reason
            plt.show()
            print(board.history)

def View_history(history):
    for idx,action in enumerate(history):
        arena.doAction(action)
        fig, ax1 = display_qzero(Stat = None, id = idx)
        draw_board(ax1, [arena.a,arena.b], np.array(arena.blocks_vertical), np.array(arena.blocks_horizon), None)
        plt.pause(0.05) #is necessary for the plot to update for some reason
        plt.show()
        print(arena.history)    
    pass      

def View_board(NNinp):
    for i in range(9):
        for j in range(9) :
            if NNinp[0][0][i][j] :
                a_loc = [i,j]
            if NNinp[0][1][i][j] :
                b_loc = [i,j]
    fig, ax1 = display_qzero(Stat = None, id = 0)
    draw_board(ax1, [a_loc,b_loc], NNinp[0,3,:8,:8], NNinp[0,2,:8,:8], None)    
    plt.pause(0.05) #is necessary for the plot to update for some reason
    plt.show()    
    pass

def plot_board(ax, NNinp):
    """
    plot board in matplotlib axis
    """
    for i in range(9):
        for j in range(9) :
            if NNinp[0][0][i][j] :
                a_loc = [i,j]
            if NNinp[0][1][i][j] :
                b_loc = [i,j]
    draw_board(ax, [a_loc,b_loc], NNinp[0,3,:8,:8], NNinp[0,2,:8,:8], None)    
    plt.pause(0.05) #is necessary for the plot to update for some reason
    plt.show()    
    pass    
