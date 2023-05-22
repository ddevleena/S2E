import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import utils_LL
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


explanation_list = ["moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, conserves side fuel usage", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, conserves main fuel usage", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left leg contact ground, conserves main fuel usage", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left leg contact ground", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets right leg contact ground, conserves main fuel", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left leg contact ground, conserves main fuel", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left leg contact ground, conserves side fuel", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets right leg contact ground", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets right leg contact ground, conserves side fuel", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left and right leg contact ground, conserves side fuel", "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left and right leg contact ground", "encourages lander to land"]
encoded_explanation_list = []

confusion_matrix = {
    0: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
    1: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
    2: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
    3: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
    4: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
    5: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
    6: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
    7: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
    8: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
    9: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
    10: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
    11: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
    12: { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0},
}

    
def test(PATH, x_test_tensor, y_test_tensor, vocab):
    model = torch.load(PATH)
    model.eval()
    count_ignore = 0
    acc_count =0
    maxLength=30
    encoded_explanation_list = train_test_utils.encode_explanation_list(vocab, maxLength, explanation_list)
    
    explanation_predictions = []
    explanation_predictions = []
    predicted_explanation_embeddings = []
    predicted_state_embeddings = []
    gt_dict = {0: 0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}  
    
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
        x_prevS1 = x_test_tensor[i,0,:].unsqueeze(0)
        x_currS = x_test_tensor[i,1,:].unsqueeze(0)
        x_gameOver = x_test_tensor[i,2,:].unsqueeze(0)
        x_explanation = x_test_tensor[i,3,:].unsqueeze(0)
        
        l2_norm_arr = []
        index_orig = 0
        total_state_embeddings = []
        total_explanation_embeddings = []
        for j, j_explanation in enumerate(explanation_list):
            get_encoded_j_explanation = data_utils.encoding(data_utils.tokenize_explanations([j_explanation]), vocab, maxLength)
            #sample a possible explanation arr to test with the given state
            new_explanation_x = torch.tensor(get_encoded_j_explanation.copy())
            gt_explanation = x_test_tensor[i,3,0:30].tolist()
            if(new_explanation_x.tolist() == gt_explanation):
                index_orig = j #saving which index in the explanation_list corresponds to the gt_explanation 
            mask_prevS1, mask_currS, mask_gameOver, mask_explanation = train_test_utils.get_masks_LL(x_prevS1, x_currS, x_gameOver, new_explanation_x)

            if USE_CUDA:
                x_prevS1 = x_prevS1.cuda()
                x_currS = x_currS.cuda()
                x_gameOver = x_gameOver.cuda()
                y_test_tensor = y_test_tensor.cuda()
                x_explanation = x_explanation.cuda()
                mask_prevS1 = mask_prevS1.cuda()
                mask_currS = mask_currS.cuda()
                mask_gameOver = mask_gameOver.cuda()
                mask_explanation = mask_explanation.cuda()
                
                new_explanation_x = new_explanation_x.type(torch.FloatTensor).cuda() #should be an tensor of vocab indexes that make up the explanation
                            
            with torch.no_grad():
                state_embed, explanation_embed = model.forward(x_prevS1, x_currS, x_gameOver, new_explanation_x, mask_prevS1,  mask_currS, mask_gameOver, mask_explanation)
                total_state_embeddings.append(state_embed.cpu().detach().numpy())
                total_explanation_embeddings.append(explanation_embed.cpu().detach().numpy())
                   
            #calculate the l2 norm difference between the state and concept_embed using 
            difference =state_embed - explanation_embed
            l2_norm = torch.linalg.norm(difference, dim=1, ord=2)           
            l2_norm_arr.append(l2_norm.cpu().item())

        min_values = sorted(range(len(l2_norm_arr)), key = lambda sub: l2_norm_arr[sub])[:3]
        predicted_index = np.argmin(np.array(l2_norm_arr))
        
        gt_key = train_test_utils.translate_to_key(x_test_tensor[i,3,0:30].tolist(), vocab, explanation_list, maxLength, "LL")
        key_at1 = train_test_utils.translate_to_key(encoded_explanation_list[min_values[0]][0].tolist(), vocab, explanation_list, maxLength, "LL")
        key_at2 = train_test_utils.translate_to_key(encoded_explanation_list[min_values[1]][0].tolist(), vocab, explanation_list, maxLength, "LL")
        key_at3 = train_test_utils.translate_to_key(encoded_explanation_list[min_values[2]][0].tolist(), vocab, explanation_list, maxLength, "LL")
        
        gt_dict[gt_key]+=1
        confusion_matrix[gt_key][key_at1] += 1
        if gt_key == key_at1:
            recall_at_1 += 1
        if gt_key == key_at1 or gt_key == key_at2:
            recall_at_2 += 1
        if gt_key == key_at1 or gt_key == key_at2 or gt_key == key_at3:
            recall_at_3 += 1
                
        #FOR TSNE compliation
        explanation_predictions.append(train_test_utils.translate_explanation_LL(explanation_list[predicted_index]))     
        predicted_explanation_embeddings.append(total_explanation_embeddings[predicted_index]) 
        predicted_state_embeddings.append(total_state_embeddings[predicted_index])
        
        key = train_test_utils.translate_to_key(x_test_tensor[i,3,0:30].tolist(), vocab, explanation_list, maxLength, "LL")
        pred_key = train_test_utils.translate_to_key(encoded_explanation_list[predicted_index][0].tolist(), vocab, explanation_list, maxLength, "LL")
        gt_dict[key]+=1
        
        confusion_matrix[key][pred_key] += 1
                
    print("concept breakdown: " + str(gt_dict))
    print("\n")
    print("--- Test Result ---")
    print("recall@1: " + str((recall_at_1/float(x_test_tensor.size(0)-count_ignore))*100))
    print("recall@2: " + str((recall_at_2/float(x_test_tensor.size(0)-count_ignore))*100))
    print("recall@3: " + str((recall_at_3/float(x_test_tensor.size(0)-count_ignore))*100))
                 
    return predicted_state_embeddings, predicted_explanation_embeddings, explanation_predictions
    

if __name__ == "__main__":
    
    model_folder_name = "test_LL/"
    
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-t", "--tsne", default=False, help="boolean for generate tsne")
    argParser.add_argument("-c", "--cm", default=True, help="plot confusion matrix")
    args = argParser.parse_args()
    
    training_loader, validation_loader, test_loader, x_test, y_test, vocab = get_data_LL(batch_size=2)
    
  
    model_files = os.listdir(model_folder_name)
    for file_name in model_files:
        pred_state_embeds, pred_exp_embeds, pred_exps = test(str(model_folder_name) + str(file_name), x_test, y_test, vocab)    
        if(args.tsne):
            print("--- Generating TSNE ---")
            train_test_utils.make_TSNE(pred_state_embeds, pred_exp_embeds, pred_exps, "LL")
            
        if(args.cm):
            print("--- Generating Confusion Matrix ---")
            train_test_utils.make_CM(confusion_matrix, "LL")

       
        
        
   