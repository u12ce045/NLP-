# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 16:56:44 2021

@author: avi00
"""

import torch
from torch import nn
import pickle
import streamlit as st
import spacy
device=torch.device('cpu')
import numpy as np
#%%
class RNN(nn.Module):
    def __init__(self,vocab_size,batch_size,emd_dim,hidden_dim,out_dim,n_layers):
        super(RNN,self).__init__()
        self.n_layers=n_layers
        self.batch_size=batch_size
        self.hidden_dim=hidden_dim
        self.embedding=nn.Embedding(vocab_size,emd_dim)
        self.gru=nn.GRU(emd_dim,hidden_dim,num_layers=n_layers,batch_first=True)
        self.fc=nn.Linear(hidden_dim,out_dim)
        self.sigmoid=nn.Sigmoid()
    def _init_hidden(self,bs):
        return torch.zeros(self.n_layers,bs,self.hidden_dim,device=device)
    def forward(self,txt,txt_lengths):
        embs=self.embedding(txt)
        packed_embs = nn.utils.rnn.pack_padded_sequence(embs, txt_lengths,batch_first=True)
        hidden=self._init_hidden(txt.size(0))
        packed_output,hidden=self.gru(packed_embs,hidden)
        outputs=self.sigmoid(self.fc(hidden))
        return outputs.squeeze()
#%%
spacy_en=spacy.load("en_core_web_sm",disable=['parser','tagger'])
def tokenizer(text):
    return [i.text for i in spacy_en.tokenizer(text) if not i.is_punct and not i.is_stop]  
with open('D:/Projects/tokenizer.pickle','rb') as f:
    stoi=pickle.load(f)
with open('D:/Projects/vocabs.pickle','rb') as f:
    vocabs=pickle.load(f)
    
#%% 
vocab_size=len(vocabs)
emd_dim=100
batch_size=64
hidden_dim=32
out_dim=1
n_layers=1
model=RNN(vocab_size,batch_size,emd_dim,hidden_dim,out_dim,n_layers).to(device)
model.load_state_dict(torch.load('D:/Projects/model_weights.pt'))
model.eval()
#%%
def predict(sentence):
    tokenized=tokenizer(sentence)
    idx=[stoi[s] for s in tokenized]
    length=[len(idx)]
    inputs=torch.LongTensor(idx).to(device)
    inputs=inputs.unsqueeze(1).T
    text_length=torch.LongTensor(length)
    predictions=model(inputs,text_length)
    if np.round(predictions.item())==1.0:
        sentiment='Positive sentiment'
    else:
        sentiment='Negative sentiment'
    return sentiment
#%%    
def run():
    st.title('SENTIMENT ANALYSIS -GRU Model')
    html_temp="""   """
    st.markdown(html_temp)
    review=st.text_input("Enter the review")
    sentiment=""
 
    if st.button('Prdict sentiment'):
        sentiment=predict(review)
    st.success("The sentiment predicted by a model is  {} ".format(sentiment))  
#%%
if __name__=='__main__':
    run()
#%% 
    
