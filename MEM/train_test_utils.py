import numpy as np 
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter
from IPython.core.debugger import set_trace
import random as rand
import spacy
import data_utils
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from matplotlib import cm
import seaborn as sn
import colorcet as cc


cm_mappings_LL = {
    0:"p_v_t_mf",
    1:"p_v_t_lf",
    2:"p_v_t",
    3:"p_v_t_ll_lf",
    4:"p_v_t_ll",
    5:"p_v_t_rl_lf",
    6:"p_v_t_ll_rl_lf",
    7:"p_v_t_ll_mf",
    8: "p_v_t_rl",
    9:"p_v_t_rl_mf",
    10:"p_v_t_ll_rl_mf",
    11:"p_v_t_ll_rl",
    12: "l"
}

dict_concepts_explanations_LL = {
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, conserves side fuel usage":0, 
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, conserves main fuel usage":1,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander":2,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left leg contact ground, conserves main fuel usage":3,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left leg contact ground":4,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets right leg contact ground, conserves main fuel":5,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left leg contact ground, conserves main fuel":6,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left leg contact ground, conserves side fuel":7,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets right leg contact ground":8,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets right leg contact ground, conserves side fuel":9,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left and right leg contact ground, conserves side fuel":10,
    "moves lander closer to the center, decreases lander speed to avoid crashing, decreases tilt of lander, lets left and right leg contact ground":11,
    "encourages lander to land":12
}


def get_masks_LL(x_prevS1, x_currS, x_gameOver, x_explanation):
    mask_prevS1 = x_prevS1.ne(99)
    mask_currS = x_currS.ne(99)
    mask_gameOver = x_gameOver.ne(99)
    mask_explanation = x_explanation.ne(99)
    
    return mask_prevS1, mask_currS, mask_gameOver, mask_explanation

def get_masks_C4(x_prevBoard,x_currBoard,x_gameOver, x_player, x_explanation):
    mask_prevB = x_prevBoard.ne(99)
    mask_currB = x_currBoard.ne(99)
    mask_gameOver = x_gameOver.ne(99)
    mask_player = x_player.ne(99)
    mask_explanation = x_explanation.ne(99)
    
    return mask_prevB, mask_currB, mask_gameOver, mask_player, mask_explanation

def encode_explanation_list(vocab, maxLength, explanation_list):
    encoded_list = []
    for explanation in explanation_list:
        encoded = data_utils.encoding(data_utils.tokenize_explanations([explanation]), vocab, maxLength)
        encoded_list.append(encoded)
    return encoded_list

def translate_to_key(arr, vocab, explanation_list, maxLength, domain):
    #first encode the explanation_list 
    key = "" 
    for explanation in explanation_list:
        encoded = data_utils.encoding(data_utils.tokenize_explanations([explanation]), vocab, maxLength)
        if(encoded[0].tolist() == arr):
            if(domain == "C4"):
                key = translate_explanation_C4(explanation)
            else:
                key = translate_explanation_LL(explanation)
    return key


def get_misclassified(x_gt, explanation_pred, vocab, explanation_list, maxLength, domain):
    which_one = translate_to_key(x_gt, vocab, explanation_list, maxLength, domain)
    with_what = translate_to_key(explanation_pred, vocab, explanation_list, maxLength, domain)
    return which_one, with_what


def translate_explanation_LL(arr):
    class_num = dict_concepts_explanations_LL[arr]
    return class_num

def translate_explanation_C4(arr):
    if(arr == "a generic move not tied to a strategy"):
        return "null"
    elif(arr == "blocks an opponent win and provides center dominance"):
        return "BW_CD"
    elif(arr == "results in a three in a row and provides center dominance"):
        return "3IR_CD"
    elif(arr == "blocks an opponent win and results in a three in a row"):
        return "BW_3IR"
    elif(arr == "results in a three in a row but is blocked from a win"):
        return "3IR_Blocked"
    elif(arr == "leads to a three in a row"):
        return "3IR"
    elif(arr == "provides center dominance"):
        return "CD"
    elif(arr == "blocks an opponent win"):
        return "BW"
    elif(arr == "leads to a win"):
        return "W"
    
def make_CM(dict_, domain):
    
    keys_ = list(dict_.keys())
    arr2 = []
    for i in keys_:
        class_sum = float(sum(dict_[i].values()))
        class_arr = []

        for j in keys_:
            calc_ = dict_[i][j] / class_sum

            class_arr.append(calc_)

        arr2.append(class_arr)
    if(domain == "C4"):
        labels = ['{BW}', '{BW,CD}', '{BW,3IR}','{3IR}', '{3IR,CD}', '{3IR_BL}', '{W}', '{CD}', '{NULL}']
        df_cm2 = pd.DataFrame(np.array(arr2)*100, labels, labels)
        fig, ax = plt.subplots(figsize=(12,12)) 
        sn.set(font_scale=1.0) # for label size
        sn.heatmap(df_cm2, annot=True, annot_kws={"size": 26}, cmap="Greens", fmt='.1f') # font size
        plt.xticks(rotation=45, fontsize=28, horizontalalignment="right", verticalalignment="top", rotation_mode="anchor")
        plt.yticks(rotation=45, fontsize=28, horizontalalignment="right", verticalalignment="bottom", rotation_mode="anchor")
        # plt.show()
        plt.tight_layout()
        plt.savefig("CM-COnnect4-TestSet_Recall1.jpg")
    else:
        labels = [cm_mappings_LL[k] for k in keys_]
        df_cm = pd.DataFrame(np.array(arr2)*100, labels, labels)
        fig, ax = plt.subplots(figsize=(12,12)) 
        sn.set(font_scale=1.0) # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, cmap="Greens", fmt='.1f') # font size
        # plt.xticks(rotation=45, fontsize=14)
        # plt.yticks(rotation=45, fontsize=14)
        # plt.show()
        plt.tight_layout()
        plt.savefig("CM-Lunarlander-TestSet_Recall1.jpg")
    
    
def make_TSNE(state_embeddings, explanation_embeddings, explanation_predictions, domain):
   
    tsne = TSNE(2, verbose=1)
    new_state_embed = []

    for list_ in state_embeddings:
        new_state_embed.append(list_[0])

    tsne_proj = tsne.fit_transform(new_state_embed)
    if(domain == "C4"):
        palette = sn.color_palette(cc.glasbey, 9)
    else:
        palette = sn.color_palette(cc.glasbey, 13)
    plt.figure(figsize=(16,10))
    sn.scatterplot(tsne_proj[:,0], tsne_proj[:,1], hue=explanation_predictions, legend='full', palette=palette)
    if(domain == "C4"):
        plt.savefig("tsne-C4.png")
    else:
        plt.savefig("tsne-LL.png")