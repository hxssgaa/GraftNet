import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from util import use_cuda, read_padded, sparse_bmm
from transformers import *


VERY_NEG_NUMBER = -100000000000


class RelReasoner(nn.Module):
    def __init__(self, pretrained_word_embedding_file, pretrained_relation_emb_file, num_relation, num_entity, num_word, num_hop,
                 entity_dim, word_dim, lstm_dropout, use_inverse_relation):
        super(RelReasoner, self).__init__()

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
        self.hidden1 = nn.Linear(in_features=word_dim, out_features=entity_dim)

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
        # self.max_pooling = nn.MaxPool1d()
        # self.bert_model = use_cuda(BertModel.from_pretrained('bert-base-uncased'))

        # self.node_linear1 = nn.Linear(in_features=entity_dim, out_features=entity_dim // 2)
        # self.node_linear2 = nn.Linear(in_features=entity_dim // 2, out_features=entity_dim // 4)
        # self.node_linear3 = nn.Linear(in_features=entity_dim // 4, out_features=self.num_relation)

    def forward(self, batch):
        query_text, relation_text, local_kb_rel_path_rels, answer_dist = batch

        batch_size, max_rel_paths = local_kb_rel_path_rels.shape

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
        # # encode query
        # query_node_emb = self.bert_model(query_text)[0]
        #
        # local_rel_emb = self.bert_model(relation_text)[0]
        query_word_emb = self.word_embedding(query_text)  # batch_size, max_query_word, word_dim
        query_hidden_emb, (query_node_emb, _) = self.node_encoder(self.lstm_drop(query_word_emb),
                                                                  self.init_hidden(1, batch_size,
                                                                                   self.entity_dim))  # 1, batch_size, entity_dim
        query_node_emb = query_node_emb.squeeze(dim=0).unsqueeze(dim=1)  # batch_size, 1, entity_dim
        query_node_emb = query_node_emb.transpose(1, 2)
        # query_hidden_emb = query_hidden_emb.transpose(1, 2)

        # rel_word_emb = self.word_embedding(relation_text)
        # _, (rel_hidden_emb, _) = self.relation_encoder(self.lstm_drop(rel_word_emb),
        #                                                self.init_hidden(1, self.num_relation,
        #                                                                 self.entity_dim))
        # rel_hidden_emb = torch.cat([rel_hidden_emb] * batch_size)

        # # load relation embedding
        # local_kb_rel_path_rels_v = local_kb_rel_path_rels.view(
        #     (local_kb_rel_path_rels.shape[0], local_kb_rel_path_rels.shape[1] * local_kb_rel_path_rels.shape[2]))
        # local_rel_emb = self.relation_embedding(local_kb_rel_path_rels_v)  # batch_size, max_fact, 2 * word_dim
        #
        # local_rel_emb = local_rel_emb.view(local_kb_rel_path_rels.shape[0] * local_kb_rel_path_rels.shape[1], local_kb_rel_path_rels.shape[2], -1)
        # _, (local_rel_hidden_emb, _) = self.relation_encoder(self.lstm_drop(local_rel_emb), self.init_hidden(1, local_rel_emb.shape[0], self.entity_dim))
        # #local_rel_emb = self.relation_linear(local_rel_emb)
        # # local_rel_emb = self.relu(self.hidden1(self.relu(local_rel_emb)))
        # local_rel_hidden_emb = local_rel_hidden_emb.squeeze(0).view(local_kb_rel_path_rels.shape[0], local_kb_rel_path_rels.shape[1], -1)

        # Attention
        div = float(np.sqrt(self.entity_dim))
        local_rel_emb = self.relation_embedding(local_kb_rel_path_rels)
        # rel2query_sim = torch.bmm(query_hidden_emb, local_rel_emb.transpose(1, 2)) / div
        # rel2query_sim = self.softmax_d1(rel2query_sim + (1 - query_mask.unsqueeze(dim=2)) * VERY_NEG_NUMBER)
        # rel2query_att = torch.sum(rel2query_sim.unsqueeze(dim=3) * query_hidden_emb.unsqueeze(dim=2), dim=1)
        # W = torch.sum(rel2query_att * local_rel_emb, dim=2) / div  # batch_size, max_fact
        # W_max = torch.max(W, dim=1, keepdim=True)[0]  # batch_size, 1
        # W_tilde = torch.exp(W - W_max)
        # local_rel_emb = local_rel_emb * W_tilde.unsqueeze(dim=2)

        # Linear classifier
        # query_node_emb = query_node_emb.squeeze(2)
        # query_node_emb = self.relu(self.node_linear1(query_node_emb))
        # query_node_emb = self.relu(self.node_linear2(query_node_emb))
        # rel_score = self.relu(self.node_linear3(query_node_emb))

        # _, (local_rel_hidden_emb, _) = self.relation_encoder(self.lstm_drop(local_rel_emb),
        #                                                      self.init_hidden(1, local_rel_emb.shape[0],
        #                                                                       self.entity_dim))
        local_rel_emb = self.relation_linear(local_rel_emb)

        # calculating score

        rel_score = local_rel_emb @ query_node_emb
        # rel_score = torch.max(rel_score / div, 2).values
        rel_score = (rel_score / div).squeeze(2)
        # rel_score = self.batch_norm(rel_score)

        loss = self.bce_loss(rel_score, answer_dist)

        # entity_score = entity_score + (1 - local_entity_mask) * VERY_NEG_NUMBER
        # pred_dist = self.sigmoid(entity_score)* local_entity_mask
        pred = torch.topk(rel_score, 2, dim=1)[1]

        return loss, pred, None  # pred, pred_dist

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return (use_cuda(torch.tensor(torch.zeros(num_layer, batch_size, hidden_size))),
                use_cuda(torch.tensor(torch.zeros(num_layer, batch_size, hidden_size))))