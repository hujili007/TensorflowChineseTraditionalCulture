from time import time
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer, BertModel
import argparse
import warnings, os
from functools import partial
import pandas as pd
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import math
from dgllife.model.gnn import GCN
import dgl
import numpy as np
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm

class DTIPredictor(nn.Module):
    def __init__(self):
        super(DTIPredictor, self).__init__()
        self.drug_in_feats = 75
        self.drug_embedding = 128
        self.drug_hidden_feats = [128, 128, 128]    
        self.drug_padding = True
        self.drug_extractor = MolecularGCN(in_feats=self.drug_in_feats, dim_embedding=self.drug_embedding,
                                           padding=self.drug_padding,
                                           hidden_feats=self.drug_hidden_feats)
        self.protein_extractor = SequenceBert()
        self.protein_dim = 128
        self.bcn = weight_norm(
            BANLayer(v_dim = self.drug_hidden_feats[-1], q_dim= self.protein_dim, h_dim = 256, h_out = 2),
            name='h_mat', dim=None)
        # 输出类别
        self.mlp_classifier = MLPDecoder(256, 512, 128, binary=1)

    def forward(self, drug_graph, input_ids, attention_mask, mode="train"):
        v_d = self.drug_extractor(drug_graph)
        v_p = self.protein_extractor(input_ids, attention_mask)
        f, att = self.bcn(v_d, v_p)
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att


class SequenceBert(nn.Module):
    def __init__(self, num_layers_to_finetune=1, output_dim=128):
        super(SequenceBert, self).__init__()
        self.bert = BertModel.from_pretrained('./prot_bert')
        self.bert_dim = self.bert.config.hidden_size

        # 维度转换层
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.bert_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

        # 冻结所有参数
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # 解冻最后num_layers_to_finetune层(当前1层)
        # for layer in self.bert.encoder.layer[-num_layers_to_finetune:]:
        #     for param in layer.parameters():
        #         param.requires_grad = True

        #  # 解冻LayerNorm和Pooler
        # for param in self.bert.pooler.parameters():
        #     param.requires_grad = True
        # for param in self.bert.encoder.layer[-1].output.LayerNorm.parameters():
        #     param.requires_grad = True
    
    def forward(self, input_ids, attention_mask):
        protein_outputs = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
        )
         # 使用所有token的输出
        sequence_output = protein_outputs.last_hidden_state  
        sequence_output = self.dim_reduction(sequence_output)  # [batch_size, seq_len, 128]
        
        return sequence_output  # 仍然是 [batch_size, seq_len, hidden_size]
    
    def count_parameters(self):
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
         # 添加类型检查
        if not hasattr(batch_graph, 'ndata'):
            raise TypeError("Expected DGLGraph object, got {}".format(type(batch_graph)))
        
        # 添加特征检查
        if 'h' not in batch_graph.ndata:
            raise KeyError("Graph does not contain node features 'h'")
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats
    

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x