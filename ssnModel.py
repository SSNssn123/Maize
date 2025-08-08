import torch
from torch import nn
import math

 
class SE(nn.Module):
    def __init__(self,in_channel,reduction=16):
        super(SE, self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out = self.fc(x)
        return out*x
   

class SpectralAttentionModule(nn.Module):
    def __init__(self, channels):
        super(SpectralAttentionModule, self).__init__()
        self.attention1 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU()
        )
        self.attention2 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        attn1_weights = self.attention1(x)
        attn2_weights = self.attention2(x)
        return attn1_weights * attn2_weights + x


class ChannelAttentionModule(nn.Module):
    def __init__(self, channels):
        super(ChannelAttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attn_weights = self.attention(x)
        return x * attn_weights + x


class HuEtAl(nn.Module):
    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels) 
            x = self.pool(self.conv(x)) 
        return x.numel() 

    def __init__(self, input_channels, n_classes, kernel_size=None, pool_size=None):
        super(HuEtAl, self).__init__()
        if kernel_size is None:
            kernel_size = math.ceil(input_channels / 9)
        if pool_size is None:
            pool_size = math.ceil(kernel_size / 5)
        
        self.input_channels = input_channels

        self.conv = nn.Conv1d(1, 20, kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.features_size = self._get_final_flattened_size()
        self.fc1 = nn.Linear(self.features_size, 100)
        self.fc2 = nn.Linear(100, n_classes)

    def forward(self, x):  
        x = x.unsqueeze(1) 
        x = torch.tanh(self.conv(x)) 
        x = torch.tanh(self.pool(x) )
        x = x.view(-1, self.features_size) 
        x = self.fc1(x) 
        x = torch.tanh(x) 
        # x = self.fc2(x) 
        return x


class CSAMCNN(nn.Module):
    def __init__(self, mha_dim, n_classes):
        super(CSAMCNN, self).__init__()
        self.mha_dim = mha_dim
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(32 * (mha_dim//2), 400)  
        self.fc1 = nn.Linear(32 * 96, 400)    
        self.fc2 = nn.Linear(400, n_classes) 

    def forward(self, x):             
        x = x.unsqueeze(1)              
        x = torch.relu(self.conv1(x))  
        x = self.pool(x)                
        x = torch.relu(self.conv2(x))   
        x = self.pool(x)               
        x = torch.relu(self.conv3(x))   
        # x = x.view(-1, 32 * (self.mha_dim//2))        
        x = x.view(-1, 32 * 96)
        x = torch.relu(self.fc1(x))    
        # out = self.fc2(x)           
        return x
           

class Multi_Branch(nn.Module):
    def __init__(self, x_channels, cA_channels, cD_channels, cn_dim, nvn_dim, mha_dim, n_classes):
        super(Multi_Branch, self).__init__()

        self.x = HuEtAl(x_channels, n_classes)
        self.cA = HuEtAl(cA_channels, n_classes)
        self.cD = HuEtAl(cD_channels, n_classes)
        self.se = SE(mha_dim)
        self.sam = SpectralAttentionModule(204)
        self.cam = ChannelAttentionModule(mha_dim)
        self.fc = nn.Linear(300, mha_dim)
        
        self.parameter_fc = nn.Sequential(
            nn.Linear(cn_dim, mha_dim*2),
            # nn.BatchNorm1d(mha_dim*2),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(mha_dim*2, mha_dim) 
        ) 
        
        self.vegetation_fc = nn.Sequential(
            nn.Linear(nvn_dim, mha_dim*2),
            # nn.BatchNorm1d(mha_dim*2),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(mha_dim*2, mha_dim)  
        ) 
        
        self.multihead_attn = nn.MultiheadAttention(mha_dim, 4)
        
        self.model = CSAMCNN(mha_dim, n_classes)
        self.cAD = HuEtAl(800, n_classes) 
        self.fc1 = nn.Linear(800, 100)  
        self.fc2 = nn.Linear(100, n_classes)  
        
    def forward(self, x, x1, cA, cD, C, N):
        output_x = self.sam(x.unsqueeze(1).transpose(1, 2))
        output_x = output_x.transpose(1, 2).squeeze(1)
        output_x = self.x(output_x)
        output_cA = self.cA(cA)
        output_cD = self.cD(cD)

        spectral = torch.cat([output_cA, output_x, output_cD], axis=1) 
        spectral = self.fc(spectral)

        output_CN = torch.cat([C, N], dim=-1)  
        parameter = self.parameter_fc(output_CN)    
        
        vegetation = self.vegetation_fc(x1)   
        vegetation = self.se(vegetation)
        
        query1 = parameter.unsqueeze(1).transpose(0, 1)  
        query2 = vegetation.unsqueeze(1).transpose(0, 1)  
        key = spectral.unsqueeze(1).transpose(0, 1)  
        value = spectral.unsqueeze(1).transpose(0, 1)  
                
        attn_output1, attn_weights1 = self.multihead_attn(query1, key, value)
        attn_output1 = attn_output1.transpose(0, 1).squeeze(1) 
        attn_output2, attn_weights2 = self.multihead_attn(query2, key, value)
        attn_output2 = attn_output2.transpose(0, 1).squeeze(1)
        
        spectral, weight_cam = self.cam(spectral.unsqueeze(1).transpose(1, 2))
        spectral = spectral.transpose(1, 2).squeeze(1)
        concatenate1 = torch.cat([parameter, spectral, attn_output1], dim=1)
        concatenate2 = torch.cat([vegetation, spectral, attn_output2], dim=1)

        output1 = self.model(concatenate1)
        output2 = self.model(concatenate2)
        concatenate = torch.cat([output1, output2], dim=1)
        output = torch.relu(self.fc1(concatenate))
        output = self.fc2(output)
        return output
    
    
