'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
from pip import main
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class GlobalPooling(nn.Module):
    def __init__(self,):
        super(GlobalPooling, self).__init__()
        self.b_avg = 9
        self.sigma = 0

    def forward(self, x, value_head=False):
        mean1 = torch.mean(torch.mean(x,dim=2),dim=2)
        mean2 = mean1*((x.shape[2]-self.b_avg)/10)
        if value_head:
            max = mean1*(((x.shape[2]-self.b_avg)**2-self.sigma**2)/100)
        else:
            max, _ = torch.max(x,dim=2)
            max, _ = torch.max(max,dim=2)
        return mean1, mean2, max

class GlobalPoolingBias(nn.Module):
    def __init__(self,cX, cG):
        super(GlobalPoolingBias, self).__init__()
        self.bn = nn.BatchNorm2d(cG)
        self.GP = GlobalPooling()
        self.fc = nn.Linear(3*cG, cX)
    
    def forward(self, X, G):
        '''
        X: bxbxcX
        G: bxbxcG
        '''
        G_out = F.relu(self.bn(G))
        b1, b2, b3 = self.GP(G_out)
        bias = torch.cat((b1,b2,b3),dim=1)
        out = self.fc(bias)
        X += out.unsqueeze(-1).unsqueeze(-1)
        return X, bias

class GlobalPoolingBlock(nn.Module):
    def __init__(self, in_planes, out_channels, c_pool=32):
        super(GlobalPoolingBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.c_pool=c_pool
        self.GPB = GlobalPoolingBias(c_pool, out_channels-c_pool)
        self.bn2 = nn.BatchNorm2d(c_pool)
        self.conv2 = nn.Conv2d(c_pool, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = x
        out = self.conv1(out)
        out, _ = self.GPB(out[:,:self.c_pool].clone(),out[:,self.c_pool:].clone()) ##########################
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out += shortcut
        return out

class PolicyHead(nn.Module):
    def __init__(self, in_channels, c_head=32):
        super(PolicyHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, c_head, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, c_head, kernel_size=1, bias=False)
        self.GPB = GlobalPoolingBias(c_head, c_head)
        self.bn = nn.BatchNorm2d(c_head)

        self.final_conv1 = nn.Conv2d(c_head, 2, kernel_size=1, bias=False)      
        self.final_conv1_1 = nn.Conv2d(c_head, 2, kernel_size=1, bias=False)

        self.final_fc1 =  nn.Linear(2*9*9, 12)

        self.final_conv2 = nn.Conv2d(c_head, 2, kernel_size=1, bias=False)
        self.final_conv2_1 = nn.Conv2d(c_head, 2, kernel_size=1, bias=False)

        self.final_fc2 = nn.Linear(2*9*9, 12)

        #self.final_fc = nn.Linear(c_head, 2)
    def forward(self, x):
        P = self.conv1(x)
        G = self.conv2(x)
        out, bias = self.GPB(P,G)
        out = F.relu(self.bn(out))

        out1 = torch.nn.functional.interpolate(self.final_conv1(out),  size = (8,8), mode='bicubic', align_corners=True)
        out1 = nn.Flatten()(out1)  #vertical 64, horizontal 64  (2,9,9)
        out1_1 = self.final_fc1(nn.Flatten()(self.final_conv1_1(out)))  # move 12

        policy_pred = torch.cat([out1,out1_1 ],dim = 1)
        policy_pred = F.softmax(policy_pred, dim=1)

        out2 = torch.nn.functional.interpolate(self.final_conv2(out),  size = (8,8), mode='bicubic', align_corners=True)
        out2 = nn.Flatten()(out2)  #opp policy 
        out2_1 =self.final_fc2(nn.Flatten()(self.final_conv2_1(out)))  # move 12

        opp_pred = torch.cat([out2,out2_1 ],dim = 1)
        opp_pred = F.softmax(opp_pred, dim=1)
        
        #out2 = self.final_fc(bias)
        return policy_pred, opp_pred

class ValueHead(nn.Module):
    def __init__(self, in_channels, c_head=32, c_val=48):
        super(ValueHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, c_head, kernel_size=1, bias=False)
        self.GP = GlobalPooling()
        #game-outcome
        self.fc1 = nn.Linear(3*c_head, c_val)
        self.fc2 = nn.Linear(c_val, 1)
        
        #final score distribution
        self.fc3 = nn.Linear(3*c_head, c_val)
        self.scaling_component = nn.Linear(c_val, 1)
        self.possible_dist = [-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15, 16, 17, 18, 19, 20]  #[-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.fc4 = nn.Linear(3*c_head+1, c_val)
        self.fc5 = nn.Linear(c_val, 1)

    def forward(self, x):
        V = self.conv1(x)
        b1,b2,b3= self.GP(V)
        V_pooled = torch.cat((b1,b2,b3),dim=1)
        game_out = F.relu(self.fc1(V_pooled))
        game_out = self.fc2(game_out)
        game_out = nn.Tanh()(game_out)

        #distance_diff
        scaling_factor = F.relu(self.fc3(V_pooled))
        scaling_factor = self.scaling_component(scaling_factor)
        logits = torch.empty(0).to(x.device)
        for d in self.possible_dist:
            V_concat = torch.cat((V_pooled, torch.full((x.shape[0],1), d).to(x.device)),dim=1)
            h = F.relu(self.fc4(V_concat))
            h = self.fc5(h)
            logits = torch.cat((logits, h),dim=1)
        logits = F.softmax(logits*scaling_factor,dim=1)
        return game_out, logits


class DistHead(nn.Module):
    def __init__(self, in_channels, c_head=32, c_val=48):
        super(DistHead, self).__init__()

        # For Dist pred
        self.conv1 = nn.Conv2d(in_channels, c_head, kernel_size=1, bias=False)
        self.GP = GlobalPooling()
        self.fc1 = nn.Linear(3*c_head, c_val)
        self.fc2 = nn.Linear(c_val, 2)

        # For Shortest Path pred
        self.convS1 = nn.Conv2d(in_channels, c_head, kernel_size=1, bias=False)
        self.convS2 = nn.Conv2d(in_channels, c_head, kernel_size=1, bias=False)      
        self.GPB = GlobalPoolingBias(c_head, c_head)
        self.bn = nn.BatchNorm2d(c_head)
        self.final_convS = nn.Conv2d(c_head, 2, kernel_size=1, bias=False)      
        self.sig = nn.Sigmoid()

    def forward(self, x):

        # Dist pred
        V = self.conv1(x)
        b1,b2,b3= self.GP(V)
        V_pooled = torch.cat((b1,b2,b3),dim=1)
        game_out = F.relu(self.fc1(V_pooled))
        game_out = self.fc2(game_out)  #  (bs, 2)

        # Shortest Path pred
        XS = self.convS1(x)
        GS = self.convS2(x)
        outS, biasS = self.GPB(XS, GS)
        outS = F.relu(self.bn(outS))
        outS = self.final_convS(outS)
        ShortPath = self.sig(outS)      # (bs, 2, 9, 9)

        return game_out, ShortPath

class DistHead2(nn.Module):
    def __init__(self, in_channels, c_head=32, c_val=48):
        super(DistHead2, self).__init__()

        # For Dist pred
        self.conv1 = nn.Conv2d(in_channels, c_head, kernel_size=1, bias=False)
        self.GP = GlobalPooling()
        self.fc1 = nn.Linear(3*c_head, c_val)
        self.fc2 = nn.Linear(c_val, 2)

        # For Shortest Path pred (6,9,9)
        self.convS1 = nn.Conv2d(in_channels, c_head, kernel_size=1, bias=False)
        self.convS2 = nn.Conv2d(in_channels, c_head, kernel_size=1, bias=False)      
        self.GPB = GlobalPoolingBias(c_head, c_head)
        self.bn = nn.BatchNorm2d(c_head)
        self.final_convS = nn.Conv2d(c_head, 6, kernel_size=1, bias=False)      
        self.sig = nn.Sigmoid()

    def forward(self, x):

        # Dist pred
        V = self.conv1(x)
        b1,b2,b3= self.GP(V)
        V_pooled = torch.cat((b1,b2,b3),dim=1)
        game_out = F.relu(self.fc1(V_pooled))
        game_out = self.fc2(game_out)  #  (bs, 2)

        # Shortest Path pred
        XS = self.convS1(x)
        GS = self.convS2(x)
        outS, biasS = self.GPB(XS, GS)
        outS = F.relu(self.bn(outS))
        outS = self.final_convS(outS)
        ShortPath = self.sig(outS)      # (bs, 6, 9, 9)

        return game_out, ShortPath


class Interpolate8t9(nn.Module):      # 미니배치에 2:4, 6:8 잘못하고 있었는데 왜 experiment1 모델이 학습이 됬지? 신기
    def __init__(self):
        super(Interpolate8t9, self).__init__()

    def forward(self, x):
        expandTo9_curr = torch.nn.functional.interpolate(x[:,2:4,:8,:8], size = (9,9), mode='bicubic', align_corners=True) #recompute_scale_factor= True) #antialias=True)
        expandTo9_prev = torch.nn.functional.interpolate(x[:,6:8,:8,:8], size = (9,9), mode='bicubic', align_corners=True) #recompute_scale_factor= True) #antialias=True)

        x[:,2:4,:,:] = expandTo9_curr
        x[:,6:8,:,:] = expandTo9_prev

        return x

class Interpolate8t9_P(nn.Module):
    def __init__(self):
        super(Interpolate8t9_P, self).__init__()
        pass

    def forward(self, x):
        expandTo9_curr = torch.nn.functional.interpolate(x[:,2:4,:8,:8], size = (9,9), mode='bicubic', align_corners=True) #recompute_scale_factor= True) #antialias=True)
        x[:,2:4,:,:] = expandTo9_curr

        return x

class ResNetb5c64(nn.Module):
    def __init__(self, input_channels=16, start_channels=64, c_pool=16, c_head=16, c_val=32):
        super(ResNetb5c64, self).__init__()

        self.interpo = Interpolate8t9()
        self.start_conv = nn.Conv2d(input_channels, start_channels, kernel_size=5, stride=1, padding=2, bias=False)

        self.ResBlock1 = PreActBlock(start_channels, start_channels, stride=1)
        self.GPBlock1 = GlobalPoolingBlock(start_channels, start_channels, c_pool)
        self.ResBlock2 = PreActBlock(start_channels, start_channels, stride=1)
        self.GPBlock2 = GlobalPoolingBlock(start_channels, start_channels, c_pool)
        self.ResBlock3 = PreActBlock(start_channels, start_channels, stride=1)

        self.last_bn = nn.BatchNorm2d(start_channels)
        self.P_head = PolicyHead(start_channels, c_head)
        self.V_head = ValueHead(start_channels, c_head, c_val)

    #def forward(self, piece_board, block_board, properties):
    def forward(self, board) :#properties):
        '''
        piece_board: 9x9x2
        block_board: 8x8x4 --> need to be upsampled
        properties: 6 scalars
        '''
        #board = torch.cat((piece_board, block_board),dim=1)
        board = self.interpo(board)
        out = self.start_conv(board)

        # n stack of residual block
        out = self.ResBlock1(out)
        out = self.GPBlock1(out)
        out = self.ResBlock2(out)
        out = self.GPBlock2(out)
        out = self.ResBlock3(out)

        out = F.relu(self.last_bn(out))

        policy_out = self.P_head(out)
        value_out = self.V_head(out)
        return policy_out, value_out

class ResNetb6c96(nn.Module):
    def __init__(self, input_channels=16, start_channels=96, c_pool=32, c_head=32, c_val=48):
        super(ResNetb6c96, self).__init__()

        self.interpo = Interpolate8t9()
        self.start_conv = nn.Conv2d(input_channels, start_channels, kernel_size=5, stride=1, padding=2, bias=False)

        self.ResBlock1 = PreActBlock(start_channels, start_channels, stride=1)
        self.ResBlock2 = PreActBlock(start_channels, start_channels, stride=1)
        self.GPBlock1 = GlobalPoolingBlock(start_channels, start_channels, c_pool)
        self.ResBlock3 = PreActBlock(start_channels, start_channels, stride=1)
        self.ResBlock4 = PreActBlock(start_channels, start_channels, stride=1)
        self.GPBlock2 = GlobalPoolingBlock(start_channels, start_channels, c_pool)

        self.last_bn = nn.BatchNorm2d(start_channels)
        self.P_head = PolicyHead(start_channels, c_head)
        self.V_head = ValueHead(start_channels, c_head, c_val)

    #def forward(self, piece_board, block_board, properties):
    def forward(self, board):
        '''
        piece_board: 9x9x2
        block_board: 8x8x4 --> need to be upsampled
        properties: 6 scalars
        '''
        #board = torch.cat((piece_board, block_board),dim=1)
        board = self.interpo(board)
        out = self.start_conv(board)

        # n stack of residual block
        out = self.ResBlock1(out)
        out = self.ResBlock2(out)
        out = self.GPBlock1(out)
        out = self.ResBlock3(out)
        out = self.ResBlock4(out)
        out = self.GPBlock2(out)

        out = F.relu(self.last_bn(out))

        policy_out = self.P_head(out)
        value_out = self.V_head(out)
        return policy_out, value_out

class ResNetb8c128(nn.Module):
    def __init__(self, input_channels=16, start_channels=128, c_pool=32, c_head=32, c_val=64):
        super(ResNetb8c128, self).__init__()
        self.interpo = Interpolate8t9()
        self.start_conv = nn.Conv2d(input_channels, start_channels, kernel_size=5, stride=1, padding=2, bias=False)

        self.ResBlock1 = PreActBlock(start_channels, start_channels, stride=1)
        self.ResBlock2 = PreActBlock(start_channels, start_channels, stride=1)
        self.ResBlock3 = PreActBlock(start_channels, start_channels, stride=1)
        self.GPBlock1 = GlobalPoolingBlock(start_channels, start_channels, c_pool)
        self.ResBlock4 = PreActBlock(start_channels, start_channels, stride=1)
        self.ResBlock5 = PreActBlock(start_channels, start_channels, stride=1)
        self.ResBlock6 = PreActBlock(start_channels, start_channels, stride=1)
        self.GPBlock2 = GlobalPoolingBlock(start_channels, start_channels, c_pool)

        self.last_bn = nn.BatchNorm2d(start_channels)
        self.P_head = PolicyHead(start_channels, c_head)
        self.V_head = ValueHead(start_channels, c_head, c_val)

    #def forward(self, piece_board, block_board, properties):
    def forward(self, board):
        '''
        piece_board: 9x9x2
        block_board: 8x8x4 --> need to be upsampled
        properties: 6 scalars
        '''
        #board = torch.cat((piece_board, block_board),dim=1)
        board = self.interpo(board)
        out = self.start_conv(board)

        # n stack of residual block
        out = self.ResBlock1(out)
        out = self.ResBlock2(out)
        out = self.ResBlock3(out)
        out = self.GPBlock1(out)
        out = self.ResBlock4(out)
        out = self.ResBlock5(out)
        out = self.ResBlock6(out)
        out = self.GPBlock2(out)

        out = F.relu(self.last_bn(out))

        policy_out = self.P_head(out)
        value_out = self.V_head(out)
        return policy_out, value_out

class ResNet5b64c_P(nn.Module):
    def __init__(self, input_channels=4, start_channels=64, c_pool=16, c_head=16, c_val=32):
        super(ResNet5b64c_P, self).__init__()

        self.interpo = Interpolate8t9_P()
        self.start_conv = nn.Conv2d(input_channels, start_channels, kernel_size=5, stride=1, padding=2, bias=False)

        self.ResBlock1 = PreActBlock(start_channels, start_channels, stride=1)
        self.GPBlock1 = GlobalPoolingBlock(start_channels, start_channels, c_pool)
        self.ResBlock2 = PreActBlock(start_channels, start_channels, stride=1)
        self.GPBlock2 = GlobalPoolingBlock(start_channels, start_channels, c_pool)
        self.ResBlock3 = PreActBlock(start_channels, start_channels, stride=1)

        self.last_bn = nn.BatchNorm2d(start_channels)
        self.D_head = DistHead(start_channels, c_head, c_val)

    #def forward(self, piece_board, block_board, properties):
    def forward(self, board) :#properties):
        '''
        piece_board: 9x9x2
        block_board: 8x8x4 --> need to be upsampled
        properties: 6 scalars
        '''
        #board = torch.cat((piece_board, block_board),dim=1)
        board = self.interpo(board)
        out = self.start_conv(board)

        # n stack of residual block
        out = self.ResBlock1(out)
        out = self.GPBlock1(out)
        out = self.ResBlock2(out)
        out = self.GPBlock2(out)
        out = self.ResBlock3(out)

        out = F.relu(self.last_bn(out))

        dist_out = self.D_head(out)
        return dist_out

class ResNet5b64c_P2(nn.Module):
    def __init__(self, input_channels=4, start_channels=64, c_pool=16, c_head=16, c_val=32):
        super(ResNet5b64c_P2, self).__init__()

        self.interpo = Interpolate8t9_P()
        self.start_conv = nn.Conv2d(input_channels, start_channels, kernel_size=5, stride=1, padding=2, bias=False)

        self.ResBlock1 = PreActBlock(start_channels, start_channels, stride=1)
        self.GPBlock1 = GlobalPoolingBlock(start_channels, start_channels, c_pool)
        self.ResBlock2 = PreActBlock(start_channels, start_channels, stride=1)
        self.GPBlock2 = GlobalPoolingBlock(start_channels, start_channels, c_pool)
        self.ResBlock3 = PreActBlock(start_channels, start_channels, stride=1)

        self.last_bn = nn.BatchNorm2d(start_channels)
        self.D_head = DistHead2(start_channels, c_head, c_val)

    #def forward(self, piece_board, block_board, properties):
    def forward(self, board) :#properties):
        '''
        piece_board: 9x9x2
        block_board: 8x8x4 --> need to be upsampled
        properties: 6 scalars
        '''
        #board = torch.cat((piece_board, block_board),dim=1)
        board = self.interpo(board)
        out = self.start_conv(board)

        # n stack of residual block
        out = self.ResBlock1(out)
        out = self.GPBlock1(out)
        out = self.ResBlock2(out)
        out = self.GPBlock2(out)
        out = self.ResBlock3(out)

        out = F.relu(self.last_bn(out))

        dist_out = self.D_head(out)
        return dist_out

# cfg = [[5,64], [6,96], [8,128]      
#[5,64] [6, 96] [8, 128]
# 5       6       8
# 64      96      128
# 16      32      32
# 16      32      32
# 32      48      64
def getModel(B,C) :
    key = f"b{B}c{C}"
    models = {'b5c64':ResNetb5c64(), 'b6c96' :ResNetb6c96(), 'b8c128' : ResNetb8c128()}
    return models[key]

def getModel_P(experiment = 1) :
    models = {1 : ResNet5b64c_P(), 2 : ResNet5b64c_P2()}
    return models[experiment]