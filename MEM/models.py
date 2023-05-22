import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from IPython.core.debugger import set_trace
import random as rand

seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
rand.seed(seed)

class MEM_LL(torch.nn.Module):
    def __init__(self,vocab_size, input_state_embed=64, hidden_state_embed=32, hidden_state_embed2=16, output_state_embed=8, exp_embed=32, output_exp_embed=8):
        super(MEM_LL, self).__init__()
        
        self.state_size = 10
        
        self.FCL_1 = torch.nn.Linear(self.state_size, input_state_embed)
        self.FCL_2 = torch.nn.Linear(input_state_embed, hidden_state_embed)
        self.hidden_2 = torch.nn.Linear((hidden_state_embed)*2+1, hidden_state_embed2)
        self.output_fc = torch.nn.Linear(hidden_state_embed2, output_state_embed) 
        
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()
        self.relu4 = torch.nn.ReLU()
        self.relu5 = torch.nn.ReLU()
        
        self.explanation_embedding = nn.Embedding(vocab_size, exp_embed, padding_idx=0)

        self.lstm = nn.LSTM(exp_embed, output_exp_embed, batch_first=True)
        
    def forward(self, x_prevS1, x_currS, x_gameOver, x_explanation, mask_prevS1, mask_currS, mask_gameOver, mask_explanation):
        
        x_cs1 = torch.masked_select(x_currS, mask_currS)
        x_cs1 = x_cs1.reshape(x_currS.size(0), self.state_size)
        
        x_ps1 = torch.masked_select(x_prevS1, mask_prevS1)
        x_ps1 = x_ps1.reshape(x_prevS1.size(0), self.state_size)
        
        x_go = torch.masked_select(x_gameOver, mask_gameOver)
        x_go = x_go.reshape(x_gameOver.size(0),1)
        
        x_e = torch.masked_select(x_explanation, mask_explanation)
        x_e = x_e.reshape(x_explanation.size(0),30)
        
        x_ps_1 = self.relu1(self.FCL_1(x_ps1))
        x_ps_2 = self.relu2(self.FCL_2(x_ps_1))
        
        x_cs_1 = self.relu3(self.FCL_1(x_cs1))
        x_cs_2 = self.relu4(self.FCL_2(x_cs_1))
        
        flatten_ps = x_ps_2.reshape((x_ps_2.size(0),-1))
        flatten_cs = x_cs_2.reshape((x_cs_2.size(0),-1))
        
        x = torch.stack((flatten_ps, flatten_cs),axis=1)
        x_flatten = x.reshape((x.size(0),-1))
        
        x = torch.cat((x_flatten, x_go), axis=-1)
        
        h1 = F.relu(self.hidden_2(x))
        state_final = self.output_fc(h1)
        
        x_e = x_e.to(torch.long)
        explanation_out1 = self.explanation_embedding(x_e)
        out_pack, (ht, ct) = self.lstm(explanation_out1)
        ht = ht.squeeze(0)
        explanation_final = self.relu5(ht)
        
        return state_final, explanation_final
    
class MEM_C4(torch.nn.Module):
    def __init__(self, vocab_size, numChannels=1, input_state_embed=64, hidden_state_embed=32, output_state_embed=16, exp_embed=32, output_exp_embed=16):
           
        super(MEM_C4, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=numChannels,out_channels=4, kernel_size=(3,3), padding='same')        
        self.conv2 = torch.nn.Conv2d(in_channels=4, out_channels=6, kernel_size=(3,3), padding='same')
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()
        self.relu4 = torch.nn.ReLU()
        self.FCL_1 = torch.nn.Linear(590, input_state_embed)
        self.hidden_fc = torch.nn.Linear(input_state_embed, hidden_state_embed)
        self.output_fc = torch.nn.Linear(hidden_state_embed, output_state_embed)         
        self.relu5 = torch.nn.ReLU()
        self.explanation_embedding = nn.Embedding(vocab_size, exp_embed, padding_idx=0)

        self.lstm = nn.LSTM(exp_embed, output_exp_embed, batch_first=True)

            
    def forward(self, x_prevBoard, x_currBoard, x_gameOver, x_player, x_explanation, mask_prevB, mask_currB, mask_gameOver, mask_player, mask_explanation):
        
        x_e = torch.masked_select(x_explanation, mask_explanation)
        x_e = x_e.reshape(x_explanation.size(0),11)
        
        x_pb = torch.masked_select(x_prevBoard, mask_prevB)
        x_pb = x_pb.reshape(x_prevBoard.size(0), 49)
        x_cb = torch.masked_select(x_currBoard, mask_currB)
        x_cb = x_cb.reshape(x_currBoard.size(0), 49)
        
        x_go = torch.masked_select(x_gameOver, mask_gameOver)
        x_go = x_go.reshape(x_gameOver.size(0),1)
        x_p = torch.masked_select(x_player, mask_player)
        x_p = x_p.reshape(x_player.size(0),1)
      
        x_pb = torch.reshape(x_pb, (x_pb.size(0),1, 7, 7))
        x_cb = torch.reshape(x_cb, (x_cb.size(0),1, 7, 7))

        x_pb_1 = self.relu1(self.conv1(x_pb))
        x_pb_2 = self.relu2(self.conv2(x_pb_1))
        
        x_cb_1 = self.relu3(self.conv1(x_cb))
        x_cb_2= self.relu4(self.conv2(x_cb_1))
        flatten_pb = x_pb_2.reshape((x_pb_2.size(0),-1))
        flatten_cb = x_cb_2.reshape((x_cb_2.size(0),-1))
        
        x = torch.stack((flatten_pb, flatten_cb),axis=1)
       
        x_flatten = x.reshape((x.size(0),-1))
        x_flatten = torch.cat((x_flatten, x_go, x_p), axis=-1)

        h1 = F.relu(self.FCL_1(x_flatten))
        h2 = F.relu(self.hidden_fc(h1))
        
        state_final = self.output_fc(h2)

        x_e = x_e.to(torch.long)
        explanation_out1 = self.explanation_embedding(x_e)
        out_pack, (ht, ct) = self.lstm(explanation_out1)
        ht = ht.squeeze(0)
        explanation_final = self.relu5(ht)
        
        return state_final, explanation_final
    


        

    
