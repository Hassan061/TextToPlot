#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 20:32:12 2022

@author: Hassan
"""

#%%
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer

from adjustText import adjust_text

from matplotlib import pyplot as plt


#%%
def read_text(text, split_text = '--------------------\n\n', split_by_lines = False):
    with open(f"./src/{text}") as f:
        if split_by_lines:
            paragraph_list  = [paragraph_paragraph_indices  for paragraph_paragraph_indices  in f.readlines() if paragraph_paragraph_indices .strip()]
        else:
            paragraph_list  = f.read().split(split_text)
            
        return paragraph_list, text
        

#%%

def add_line_after_count(paragraph_list , WORDCOUNT = 20):
    new_paragraph_list   = []
    
    for sent in paragraph_list :
        split_paragraph_paragraph_indices  = sent.split(" ")
        word_count = 0 
        new_input_list = [] 
        user_new_input = '' 
        for c in split_paragraph_paragraph_indices : 
            word_count += 1 
            new_input_list.append(c + " ") 
            if word_count == WORDCOUNT: 
                new_input_list.append('\n') 
                word_count = 0 
                
        new_sent = user_new_input.join(new_input_list)
        new_paragraph_list.append(new_sent)
    
    return new_paragraph_list 
    
#%%
def paragraph_list_split (paragraph_list, WORDCOUNT_TOSPLIT = 3000 ):

    sumwords  = 0
    paragraph_indices   = []
    for i, paragraphs in enumerate(paragraph_list):
        sumwords  += len(paragraphs)
        if sumwords  >= WORDCOUNT_TOSPLIT:
            paragraph_indices.append(i)
            sumwords  = 0
            
            
    if len(paragraph_indices) > 0:
        size = len(paragraph_list)
        
        res = [paragraph_list[i: j] for i, j in
                zip([0] + paragraph_indices, paragraph_indices + 
                ([size] if paragraph_indices[-1] != size else []))]
        
        if len(res[-1]) == 1:
            res[-1].append('END')
            
    else: 
        res = paragraph_list
        
    return res, paragraph_indices

#%%

def find_paragraph_similarity(paragraph_list,paragraph_indices ):
    scaler = MinMaxScaler()
    model = SentenceTransformer('paraphrase-mpnet-base-v2')
    
    enc_data_list = []
    if len(paragraph_indices) > 0:
        for paragraph  in paragraph_list:
            encoded_data = model.encode(paragraph, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
            scaler.fit(encoded_data)
            encoded_data = scaler.transform(encoded_data)
            enc_data_list.append(encoded_data)
    else:
        encoded_data = model.encode(paragraph_list, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        scaler.fit(encoded_data)
        encoded_data = scaler.transform(encoded_data)
        enc_data_list.append(encoded_data)
    
    two_dim_list = []
    ax_list = []
    ay_list = []
    for data in enc_data_list:
        two_dim = PCA(random_state=0).fit_transform(data)[:,:2]
        ax = two_dim[:,0].reshape(-1,1)
        ay = two_dim[:,1].reshape(-1,1)
        
        two_dim_list.append(two_dim)
        ax_list.append(ax)
        ay_list.append(ay)
        
    return ax_list, ay_list


#%%
def plot_paragraphs(paragraph_list,paragraph_indices, text, WIDTH = 40, HEIGHT = 50, FONTSIZE = 20, save = True, plot = False, isTransparent = True):
    i = 0
    text = text.split('.')[0]
    if len(paragraph_indices) > 0:
        for ax, ay, paragraphs in zip(ax_list, ay_list, paragraph_list):
            
            fig, aj = plt.subplots(figsize=(WIDTH,HEIGHT))

            plt.scatter(ax, ay, marker = '')
            
            texts = [plt.text(ax[i], ay[i], paragraphs[i], fontsize = FONTSIZE) for i in range(len(ax))]
            adjust_text(texts)
            #adjust_text(texts)
            
            # Selecting the axis-X making the bottom and top axes False.
            plt.tick_params(axis='x', which='both', bottom=False,
                            top=False, labelbottom=False)
              
            # Selecting the axis-Y making the right and left axes False
            plt.tick_params(axis='y', which='both', right=False,
                            left=False, labelleft=False)
              
            # Iterating over all the axes in the figure
            # and make the Spines Visibility as False
            for pos in ['right', 'top', 'bottom', 'left']:
                plt.gca().spines[pos].set_visible(False)
            if plot: 
                plt.figure(facecolor="white")
            if save:
                fig.savefig(f'./dst/{text}{i}.png', transparent=isTransparent)
            i+=1
    else:
        fig, aj = plt.subplots(figsize=(WIDTH,HEIGHT))
        ax = ax_list[0]
        ay = ay_list[0]
        #plt.figure(figsize=(60,100))
        plt.scatter(ax_list[0], ay_list[0], marker = '')
        
        texts = [plt.text(ax[i], ay[i], paragraph_list[i], fontsize = FONTSIZE) for i in range(len(ax))]
        adjust_text(texts)
        #adjust_text(texts)
        
        # Selecting the axis-X making the bottom and top axes False.
        plt.tick_params(axis='x', which='both', bottom=False,
                        top=False, labelbottom=False)
          
        # Selecting the axis-Y making the right and left axes False
        plt.tick_params(axis='y', which='both', right=False,
                        left=False, labelleft=False)
          
        # Iterating over all the axes in the figure
        # and make the Spines Visibility as False
        for pos in ['right', 'top', 'bottom', 'left']:
            plt.gca().spines[pos].set_visible(False)
        
        if plot: 
            plt.figure(facecolor="white")
        if save:
            fig.savefig(f'./dst/{text}{i}.png', transparent=isTransparent)
        i+=1

#%%

if __name__ == "__main__":
    paragraph_list, text  = read_text("HowEmotionsAreMade.txt", split_by_lines=False)
    paragraph_list  = add_line_after_count(paragraph_list, WORDCOUNT=20)
    paragraph_list, paragraph_indices  = paragraph_list_split(paragraph_list, WORDCOUNT_TOSPLIT = 3000)
    ax_list, ay_list = find_paragraph_similarity(paragraph_list,paragraph_indices ) 
       
    plot_paragraphs(paragraph_list,paragraph_indices, text, plot = True, save = True, isTransparent=False)
