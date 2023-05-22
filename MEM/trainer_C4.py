import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import random as rand
from IPython.core.debugger import set_trace
from torch.utils.tensorboard import SummaryWriter
import train_test_utils

from torch.utils.data import TensorDataset, DataLoader

from models import *
from losses import loss_fn
from data import *

USE_CUDA = torch.cuda.is_available()
DEVICE=torch.device('cuda:0')

      
def run_one_epoch(writer,data_loader, model, optimizer):
    
    running_loss =0 

    for i, data in enumerate(data_loader):
        inputs, y = data
        x_prevBoard = inputs[:,0,:]
        x_currBoard = inputs[:,1,:]
        x_gameOver = inputs[:,2,:]
        x_player = inputs[:,3,:]
        x_explanation = inputs[:,4,:]
        mask_prevB, mask_currB, mask_gameOver, mask_player, mask_explanation = train_test_utils.get_masks_C4(x_prevBoard,x_currBoard, x_gameOver, x_player, x_explanation)
        
        if USE_CUDA:
            x_prevBoard = x_prevBoard.cuda()
            x_currBoard = x_currBoard.cuda()
            x_gameOver = x_gameOver.cuda()
            x_player = x_player.cuda()
            x_explanation = x_explanation.cuda()
            y = y.cuda()            
            mask_prevB = mask_prevB.cuda()
            mask_currB = mask_currB.cuda()
            mask_gameOver = mask_gameOver.cuda()
            mask_player = mask_player.cuda()
            mask_explanation = mask_explanation.cuda()
        
        optimizer.zero_grad()
        
        state_embed, explanation_embed = model.forward(x_prevBoard, x_currBoard, x_gameOver, x_player, x_explanation, mask_prevB, mask_currB, mask_gameOver, mask_player, mask_explanation)
        state_embed.cuda()
        explanation_embed.cuda()

        loss = loss_fn(state_embed, explanation_embed, y)
        
        loss.backward()
        optimizer.step()
        
        running_loss +=loss
           
    return running_loss/float(i+1)

def evaluate(writer, validation_loader, model):
    running_v_loss = 0.0
    running_v_acc = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            #get the data 
            inputs, y = data
            x_prevBoard = inputs[:,0,:]
            x_currBoard = inputs[:,1,:]
            x_gameOver = inputs[:,2,:]
            x_player = inputs[:,3,:]
            x_explanation = inputs[:,4,:]
            mask_prevB, mask_currB, mask_gameOver, mask_player, mask_explanation = train_test_utils.get_masks_C4(x_prevBoard,x_currBoard, x_gameOver, x_player, x_explanation)       

            if USE_CUDA:
                x_prevBoard = x_prevBoard.cuda()
                x_currBoard = x_currBoard.cuda()
                x_gameOver = x_gameOver.cuda()
                x_player = x_player.cuda()
                x_explanation = x_explanation.cuda()
                y = y.cuda()            
                mask_prevB = mask_prevB.cuda()
                mask_currB = mask_currB.cuda()
                mask_gameOver = mask_gameOver.cuda()
                mask_player = mask_player.cuda()
                mask_explanation = mask_explanation.cuda()

            state_embed, explanation_embed = model.forward(x_prevBoard, x_currBoard, x_gameOver, x_player, x_explanation, mask_prevB, mask_currB, mask_gameOver, mask_player, mask_explanation)
            state_embed.cuda()
            explanation_embed.cuda()
            
            v_loss = loss_fn(state_embed, explanation_embed, y)            
            running_v_loss+=v_loss
            
            valid_loss = running_v_loss/float(i+1)
   
    return valid_loss

def train(writer, training_loader,validation_loader,model, num_epochs=1, lr=0.001,batch_size=2, seed=0):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if USE_CUDA:
        model.cuda()   
    
    train_loss_arr = []
    valid_loss_arr = []
    
    best_vloss = float('inf')
    
    
    for epoch in range(num_epochs):
        print("Epoch ", epoch)
        model.train()
        train_loss = run_one_epoch(writer, training_loader,model,optimizer)
        print("Training Loss: %f" % (train_loss))
        valid_loss = evaluate(writer, validation_loader,model)
        print("Validation Loss: %f" % (valid_loss))
        writer.add_scalars("Training vs. Validation Loss", 
                          {"Training": train_loss, "Validation": valid_loss},
                          epoch+1)
        writer.flush()
        
        if(valid_loss < best_vloss):
            best_vloss = valid_loss
            model_path = "MEM-C4_seed_" +str(seed)+ ".pth"
            torch.save(model, model_path)          

if __name__ == "__main__":
    
    lr = 0.001
    batch_size = 128
    num_epochs = 10
    input_state_embed = 64
    hidden_state_embed = 32
    output_state_embed = 16 
    exp_embed = 32
    output_exp_embed = 16
    
    seeds = [42]
    for seed in seeds:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        rand.seed(seed)

        writer = SummaryWriter("tb/MEM-C4_seed_" +str(seed)+ ".pth")

        training_loader, validation_loader, test_loader, x_test, y_test, vocab = get_data_C4(batch_size=batch_size)
        
        vocab_size = len(list(vocab.keys()))

        print("data loading finished")
        print("start training MEM")

        MEM = MEM_C4(vocab_size, input_state_embed=input_state_embed, hidden_state_embed=hidden_state_embed, output_state_embed=output_state_embed, exp_embed=exp_embed, output_exp_embed=output_exp_embed)

        train(writer, training_loader, validation_loader, MEM, num_epochs=num_epochs, lr=lr,batch_size=batch_size, seed=seed)



    
    