import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import random as rand
import train_test_utils
from IPython.core.debugger import set_trace
from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import TensorDataset, DataLoader
from models import *
from losses import loss_fn
from data import *

USE_CUDA = torch.cuda.is_available()
DEVICE=torch.device('cuda:0')




def run_one_epoch(writer,data_loader, model, optimizer):
    
    running_loss =0 
    last_loss = 0

    for i, data in enumerate(data_loader):
        #get the data 
        inputs, y = data
        x_prevS1 = inputs[:,0,:]
        x_currS = inputs[:,1,:]
        x_gameOver = inputs[:,2,:]
        x_explanation = inputs[:,3,:]
        
        y.to(torch.int64)
            
        mask_prevS1, mask_currS,mask_gameOver, mask_explanation = train_test_utils.get_masks_LL(x_prevS1, x_currS, x_gameOver, x_explanation)
        
        if USE_CUDA:
            x_prevS1 = x_prevS1.cuda()
            x_currS = x_currS.cuda()
            x_gameOver = x_gameOver.cuda()
            x_explanation = x_explanation.cuda()
            y = y.cuda()
            mask_prevS1 = mask_prevS1.cuda()
            mask_currS = mask_currS.cuda()
            mask_gameOver = mask_gameOver.cuda()
            mask_explanation = mask_explanation.cuda()
        
        optimizer.zero_grad()
        state_embed, explanation_embed = model.forward(x_prevS1, x_currS, x_gameOver, x_explanation, mask_prevS1, mask_currS, mask_gameOver, mask_explanation)
        
        state_embed.cuda()
        explanation_embed.cuda()

        loss = loss_fn(state_embed, explanation_embed, y)
        
        loss.backward()
        optimizer.step()
        
        running_loss +=loss
           
    return running_loss/float(i+1)

def evaluate(writer, validation_loader,model):
    running_v_loss = 0.0
    running_v_acc = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            #get the data 
            inputs, y = data
            x_prevS1 = inputs[:,0,:]
            x_currS = inputs[:,1,:]
            x_gameOver = inputs[:,2,:]
            x_explanation = inputs[:,3,:]

            y.to(torch.int64)

            mask_prevS1, mask_currS,mask_gameOver, mask_explanation = train_test_utils.get_masks_LL(x_prevS1, x_currS, x_gameOver, x_explanation)
        
        if USE_CUDA:
            x_prevS1 = x_prevS1.cuda()
            x_currS = x_currS.cuda()
            x_gameOver = x_gameOver.cuda()
            x_explanation = x_explanation.cuda()
            y = y.cuda()
            mask_prevS1 = mask_prevS1.cuda()
            mask_currS = mask_currS.cuda()
            mask_gameOver = mask_gameOver.cuda()
            mask_explanation = mask_explanation.cuda()

            state_embed, explanation_embed = model.forward(x_prevS1, x_currS, x_gameOver, x_explanation, mask_prevS1, mask_currS, mask_gameOver, mask_explanation)
        
            state_embed.cuda()
            explanation_embed.cuda()

            v_loss = loss_fn(state_embed, explanation_embed, y)            
            running_v_loss+=v_loss
            
            valid_loss = running_v_loss/float(i+1)
   
    return valid_loss

def train(writer, training_loader,validation_loader,model, num_epochs=1, lr=0.001, batch_size=2,seed=0):
    
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
            model_path = "MEM_LL_seed_" + str(seed) + ".pth"
            torch.save(model, model_path)          

if __name__ == "__main__":
    
    lr = 0.001
    batch_size = 128
    num_epochs = 10
    input_state_embed=64
    hidden_state_embed=32
    hidden_state_embed2=16
    output_state_embed=8
    exp_embed=32
    output_exp_embed=8
    seeds = [42]

    for seed in seeds:
       
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        rand.seed(seed)
        writer = SummaryWriter("tb/MEM_LL_seed_" + str(seed) + ".pth")

        training_loader, validation_loader, test_loader,x_test, y_test, vocab = get_data_LL(batch_size=batch_size)

            
        vocab_size = len(list(vocab.keys()))
        
        print("data loading finished")
        print("start training MEM")

        MEM = MEM_LL(vocab_size, input_state_embed=input_state_embed, hidden_state_embed=hidden_state_embed, hidden_state_embed2=hidden_state_embed2, output_state_embed=output_state_embed, exp_embed=exp_embed, output_exp_embed=output_exp_embed)

        train(writer, training_loader, validation_loader, MEM, num_epochs=num_epochs, lr=lr, batch_size=batch_size, seed=seed)
                               
                    


    
    