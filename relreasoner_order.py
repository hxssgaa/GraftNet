import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from util import use_cuda, read_padded, sparse_bmm
from transformers import *


VERY_NEG_NUMBER = -100000000000


class RelOrderReasoner(nn.Module):
    def __init__(self, pretrained_word_embedding_file, pretrained_relation_emb_file, num_relation, num_entity, num_word, num_hop,
                 entity_dim, word_dim, lstm_dropout, use_inverse_relation):
        super(RelOrderReasoner, self).__init__()

        self.num_relation = num_relation
        self.num_entity = num_entity
        self.num_word = num_word
        self.entity_dim = entity_dim
        self.word_dim = word_dim

        self.relation_embedding = nn.Embedding(num_embeddings=num_relation + 1, embedding_dim=word_dim,
                                               padding_idx=num_relation)
        if pretrained_relation_emb_file is not None:
            self.relation_embedding.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(pretrained_relation_emb_file), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
            print('loaded relation_emb')
            self.relation_embedding.weight.requires_grad = False

        self.word_embedding = nn.Embedding(num_embeddings=num_word + 1, embedding_dim=word_dim, padding_idx=num_word)
        if pretrained_word_embedding_file is not None:
            self.word_embedding.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(pretrained_word_embedding_file), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
            self.word_embedding.weight.requires_grad = False
            print('load word emb')

        self.relation_linear = nn.Linear(in_features=word_dim, out_features=entity_dim)

        self.node_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim, batch_first=True, bidirectional=False)
        self.relation_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim, batch_first=True, bidirectional=False)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = use_cuda(torch.nn.BatchNorm1d(num_relation))
        self.softmax_d1 = nn.Softmax(dim=1)
        # dropout
        self.lstm_drop = nn.Dropout(p=0.0)
        # loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.relu = nn.ReLU()


    def forward(self, batch):
        query_text, relation_text, local_kb_rel_path_rels, answer_dist = batch

        batch_size, max_rel_paths, num_hop = local_kb_rel_path_rels.shape

        # numpy to tensor
        with torch.no_grad():
            query_text = use_cuda(Variable(torch.from_numpy(query_text).type('torch.LongTensor')))
            query_mask = use_cuda((query_text != self.num_word).type('torch.FloatTensor'))
            relation_text = use_cuda(Variable(torch.from_numpy(relation_text).type('torch.LongTensor')))
            relation_mask = use_cuda((relation_text != self.num_word).type('torch.FloatTensor'))
            local_kb_rel_path_rels = use_cuda(
                Variable(torch.from_numpy(local_kb_rel_path_rels).type('torch.LongTensor')))
            answer_dist = use_cuda(
                Variable(torch.from_numpy(answer_dist).type('torch.FloatTensor')))

        query_word_emb = self.word_embedding(query_text)  # batch_size, max_query_word, word_dim
        query_hidden_emb, (query_node_emb, _) = self.node_encoder(self.lstm_drop(query_word_emb),
                                                                  self.init_hidden(1, batch_size,
                                                                                   self.entity_dim))  # 1, batch_size, entity_dim
        query_node_emb = query_node_emb.squeeze(dim=0).unsqueeze(dim=1)  # batch_size, 1, entity_dim
        query_node_emb = query_node_emb.transpose(1, 2)
        div = float(np.sqrt(self.entity_dim))
        local_rel_emb = self.relation_embedding(local_kb_rel_path_rels)
        local_rel_emb = local_rel_emb.view(-1, local_rel_emb.shape[2], local_rel_emb.shape[3])
        local_rel_hidden_emb, (local_rel_node_emb, _) = self.relation_encoder(self.lstm_drop(local_rel_emb),
                                                                              self.init_hidden(1, local_rel_emb.shape[0],
                                                                                               self.entity_dim))

        local_rel_node_emb = local_rel_node_emb.squeeze(dim=0).unsqueeze(dim=1)  # batch_size, 1, entity_dim
        local_rel_node_emb = local_rel_node_emb.transpose(1, 2)
        local_rel_node_emb = local_rel_node_emb.view(batch_size, max_rel_paths, -1)
        # local_rel_emb = self.relation_linear(local_rel_node_emb)

        # calculating score

        rel_score = local_rel_node_emb @ query_node_emb
        rel_score = (rel_score / div).squeeze(2)

        loss = self.bce_loss(rel_score, answer_dist)

        pred = torch.topk(rel_score, num_hop, dim=1)[1]

        return loss, pred, None  # pred, pred_dist

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return (use_cuda(torch.tensor(torch.zeros(num_layer, batch_size, hidden_size))),
                use_cuda(torch.tensor(torch.zeros(num_layer, batch_size, hidden_size))))