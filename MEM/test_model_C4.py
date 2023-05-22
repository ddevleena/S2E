import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import data_utils
from IPython.core.debugger import set_trace
from sklearn.manifold import TSNE
from torch import linalg 
import os, sys
import random as rand
import json
import spacy
import data_utils 
import train_test_utils
import argparse


from torch.utils.data import TensorDataset, DataLoader

from models import *
from losses import loss_fn
from data import *


seed = 42

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
rand.seed(seed)

explanation_list = ["a generic move not tied to a strategy", "blocks an opponent win and provides center dominance","results in a three in a row and provides center dominance", "blocks an opponent win and results in a three in a row", "results in a three in a row but is blocked from a win", "leads to a three in a row", "provides center dominance", "blocks an opponent win", "leads to a win"]
encoded_explanation_list = []

confusion_matrix = {
    "BW": { "BW": 0, "BW_CD": 0, "BW_3IR": 0, "3IR": 0, "3IR_CD": 0, "3IR_Blocked": 0, "W": 0, "CD": 0, "null": 0 },
    "BW_CD": { "BW": 0, "BW_CD": 0, "BW_3IR": 0, "3IR": 0, "3IR_CD": 0, "3IR_Blocked": 0, "W": 0, "CD": 0, "null": 0 },
    "BW_3IR": { "BW": 0, "BW_CD": 0, "BW_3IR": 0, "3IR": 0, "3IR_CD": 0, "3IR_Blocked": 0, "W": 0, "CD": 0, "null": 0 },
    "3IR": { "BW": 0, "BW_CD": 0, "BW_3IR": 0, "3IR": 0, "3IR_CD": 0, "3IR_Blocked": 0, "W": 0, "CD": 0, "null": 0 },
    "3IR_CD": { "BW": 0, "BW_CD": 0, "BW_3IR": 0, "3IR": 0, "3IR_CD": 0, "3IR_Blocked": 0, "W": 0, "CD": 0, "null": 0 },
    "3IR_Blocked": { "BW": 0, "BW_CD": 0, "BW_3IR": 0, "3IR": 0, "3IR_CD": 0, "3IR_Blocked": 0, "W": 0, "CD": 0, "null": 0 },
    "W": { "BW": 0, "BW_CD": 0, "BW_3IR": 0, "3IR": 0, "3IR_CD": 0, "3IR_Blocked": 0, "W": 0, "CD": 0, "null": 0 },
    "CD": { "BW": 0, "BW_CD": 0, "BW_3IR": 0, "3IR": 0, "3IR_CD": 0, "3IR_Blocked": 0, "W": 0, "CD": 0, "null": 0 },
    "null": { "BW": 0, "BW_CD": 0, "BW_3IR": 0, "3IR": 0, "3IR_CD": 0, "3IR_Blocked": 0, "W": 0, "CD": 0, "null": 0 },
}


def test(PATH, x_test_tensor, y_test_tensor, vocab):
    model = torch.load(PATH)
    model.eval()
    count_ignore = 0
    acc_count =0
    maxLength=11
    encoded_explanation_list = train_test_utils.encode_explanation_list(vocab, maxLength, explanation_list)
    
    gt_dict = {"null":0, "BW_CD":0, "3IR_CD":0, "BW_3IR":0, "3IR_Blocked":0, "3IR":0, "CD":0, "W":0, "BW":0}
    
    explanation_predictions = []
    predicted_explanation_embeddings = []
    predicted_state_embeddings = []
    
    recall_at_1 = 0
    recall_at_2 = 0
    recall_at_3 = 0
        
    for i in range(x_test_tensor.size(0)):
        #only care about evaluating the data samples where the the concept arr and state are aligned
        if(y_test_tensor[i] == 1): 
            count_ignore+=1
            continue
        print("test sample: " + str(i))
        #get the original data                      
        x_prevBoard = x_test_tensor[i,0,:].unsqueeze(0)
        x_currBoard = x_test_tensor[i,1,:].unsqueeze(0)
        x_gameOver = x_test_tensor[i,2,:].unsqueeze(0)
        x_player = x_test_tensor[i,3,:].unsqueeze(0)
        x_explanation = x_test_tensor[i,4,:].unsqueeze(0)
        
        l2_norm_arr = []
        index_orig = 0
        total_state_embeddings = []
        total_explanation_embeddings = []
        for j, j_explanation in enumerate(explanation_list):
            get_encoded_j_explanation = data_utils.encoding(data_utils.tokenize_explanations([j_explanation]), vocab, maxLength)
            #sample a possible explanation arr to test with the given state
            new_explanation_x = torch.tensor(get_encoded_j_explanation.copy())
            gt_explanation = x_test_tensor[i,4,0:11].tolist()
            if(new_explanation_x.tolist() == gt_explanation):
                index_orig = j #saving which index in the explanation_list corresponds to the gt_explanation 
            mask_prevB, mask_currB, mask_gameOver, mask_player, mask_explanation = train_test_utils.get_masks_C4(x_prevBoard,x_currBoard,x_gameOver, x_player, new_explanation_x)
        
            if USE_CUDA:
                x_prevBoard = x_prevBoard.cuda()
                x_currBoard = x_currBoard.cuda()
                x_gameOver = x_gameOver.cuda()
                x_player = x_player.cuda()
                x_explanation = x_explanation.cuda()
                mask_prevB = mask_prevB.cuda()
                mask_currB = mask_currB.cuda()                
                mask_gameOver = mask_gameOver.cuda()
                mask_player = mask_player.cuda()
                mask_explanation = mask_explanation.cuda()
                
                new_explanation_x = new_explanation_x.type(torch.FloatTensor).cuda() #should be an tensor of vocab indexes that make up the explanation
                            
            with torch.no_grad():
                state_embed, explanation_embed = model.forward(x_prevBoard, x_currBoard, x_gameOver, x_player, new_explanation_x, mask_prevB, mask_currB, mask_gameOver, mask_player, mask_explanation)
                total_state_embeddings.append(state_embed.cpu().detach().numpy())
                total_explanation_embeddings.append(explanation_embed.cpu().detach().numpy())
            
            #calculate the l2 norm difference between the state and concept_embed using 
            difference =state_embed - explanation_embed
            l2_norm = torch.linalg.norm(difference, dim=1, ord=2)           
            l2_norm_arr.append(l2_norm.cpu().item())

        min_values = sorted(range(len(l2_norm_arr)), key = lambda sub: l2_norm_arr[sub])[:3]
        predicted_index = np.argmin(np.array(l2_norm_arr))
        
        gt_key = train_test_utils.translate_to_key(x_test_tensor[i,4,0:11].tolist(), vocab, explanation_list, maxLength, "C4")
        key_at1 = train_test_utils.translate_to_key(encoded_explanation_list[min_values[0]][0].tolist(), vocab, explanation_list, maxLength, "C4")
        key_at2 = train_test_utils.translate_to_key(encoded_explanation_list[min_values[1]][0].tolist(), vocab, explanation_list, maxLength, "C4")
        key_at3 = train_test_utils.translate_to_key(encoded_explanation_list[min_values[2]][0].tolist(), vocab, explanation_list, maxLength, "C4")
        
        gt_dict[gt_key]+=1
        confusion_matrix[gt_key][key_at1] += 1
        if gt_key == key_at1:
            recall_at_1 += 1
        if gt_key == key_at1 or gt_key == key_at2:
            recall_at_2 += 1
        if gt_key == key_at1 or gt_key == key_at2 or gt_key == key_at3:
            recall_at_3 += 1
        
        #FOR TSNE compliation
        explanation_predictions.append(train_test_utils.translate_explanation_C4(explanation_list[predicted_index]))     
        predicted_explanation_embeddings.append(total_explanation_embeddings[predicted_index]) 
        predicted_state_embeddings.append(total_state_embeddings[predicted_index])

    print("concept breakdown: " + str(gt_dict))
    print("\n")
    print("--- Test Results ---")
    print("recall@1: " + str((recall_at_1/float(x_test_tensor.size(0)-count_ignore))*100))
    print("recall@2: " + str((recall_at_2/float(x_test_tensor.size(0)-count_ignore))*100))
    print("recall@3: " + str((recall_at_3/float(x_test_tensor.size(0)-count_ignore))*100))
    
    return predicted_state_embeddings, predicted_explanation_embeddings, explanation_predictions
    

if __name__ == "__main__":
    
    model_folder_name = "test_C4/"
    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-t", "--tsne", default=False, help="boolean for generate tsne")
    argParser.add_argument("-cm", "--cm", default=True, help="plot confusion matrix")
    args = argParser.parse_args()
    
    training_loader, validation_loader, test_loader, x_test, y_test, vocab = get_data_C4(batch_size=2)
    
    model_files = os.listdir(model_folder_name)
    for file_name in model_files:
        pred_state_embeds, pred_exp_embeds, pred_exps = test(str(model_folder_name) + str(file_name), x_test, y_test, vocab)   
        
        if(args.tsne):
            print("--- Generating TSNE ---")
            train_test_utils.make_TSNE(pred_state_embeds, pred_exp_embeds, pred_exps, "C4")
            
        if(args.cm):
            print("--- Generating Confusion Matrix ---")
            train_test_utils.make_CM(confusion_matrix, "C4")
 


       
        
        
   