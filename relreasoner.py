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
        self.num_lstm_layer = 1
        self.train_eod = False

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
        # self.eod_linear1 = nn.Linear(in_features=entity_dim * 2, out_features=entity_dim)
        # self.eod_linear1_1 = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        # self.eod_linear2 = nn.Linear(in_features=entity_dim, out_features=100)
        # self.eod_linear3 = nn.Linear(in_features=100, out_features=1)
        # self.relation_linear2 = nn.Linear(in_features=word_dim, out_features=entity_dim)
        # self.relation_linear3 = nn.Linear(in_features=word_dim, out_features=entity_dim)
        # self.hidden1 = nn.Linear(in_features=word_dim, out_features=entity_dim)
        # self.relation_weight = nn.Linear(in_features=3, out_features=1)

        self.node_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim, num_layers=self.num_lstm_layer, batch_first=True, bidirectional=False)
        self.relation_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim, num_layers=self.num_lstm_layer, batch_first=True, bidirectional=False)
        self.seed_type_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.question_seed_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.sigmoid = nn.Sigmoid()
        # self.batch_norm = use_cuda(torch.nn.BatchNorm1d(word_dim if num_hop == 2 else 300))
        self.softmax_d1 = nn.Softmax(dim=1)
        # dropout
        self.lstm_drop = nn.Dropout(p=lstm_dropout)
        self.linear_drop = nn.Dropout(p=0.2)
        # loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.relu = nn.ReLU()
        # self.max_pooling = nn.MaxPool1d()
        # self.bert_model = use_cuda(BertModel.from_pretrained('bert-base-uncased'))

        # self.node_linear1 = nn.Linear(in_features=entity_dim, out_features=entity_dim // 2)
        # self.node_linear2 = nn.Linear(in_features=entity_dim // 2, out_features=entity_dim // 4)
        # self.node_linear3 = nn.Linear(in_features=entity_dim // 4, out_features=self.num_relation)

    def forward(self, batch):
        query_text, relation_text, local_kb_rel_path_rels, seed_entity_types, answer_dist = batch

        batch_size, max_rel_paths, num_hop = local_kb_rel_path_rels.shape
        max_rel_num = relation_text.shape[2]

        # numpy to tensor
        with torch.no_grad():
            query_text = use_cuda(Variable(torch.from_numpy(query_text).type('torch.LongTensor')))
            query_mask = use_cuda((query_text != self.num_word).type('torch.FloatTensor'))
            relation_text = use_cuda(Variable(torch.from_numpy(relation_text).type('torch.LongTensor')))
            relation_mask = use_cuda((relation_text != self.num_word).type('torch.FloatTensor'))
            seed_entity_types = use_cuda(Variable(torch.from_numpy(seed_entity_types).type('torch.LongTensor')))
            local_kb_rel_path_rels = use_cuda(
                Variable(torch.from_numpy(local_kb_rel_path_rels).type('torch.LongTensor')))
            local_kb_rel_path_rels_last = local_kb_rel_path_rels[:, :, -1]
            local_kb_rel_path_rels_mask = use_cuda((local_kb_rel_path_rels_last != self.num_relation).type('torch.FloatTensor'))
            answer_dist = use_cuda(
                Variable(torch.from_numpy(answer_dist).type('torch.FloatTensor')))
        # # encode query
        # query_node_emb = self.bert_model(query_text)[0]
        #
        # local_rel_emb = self.bert_model(relation_text)[0]
        query_word_emb = self.word_embedding(query_text)  # batch_size, max_query_word, word_dim
        query_hidden_emb, (query_node_emb, _) = self.node_encoder(self.lstm_drop(query_word_emb),
                                                                  self.init_hidden(self.num_lstm_layer, batch_size,
                                                                                   self.entity_dim))  # 1, batch_size, entity_dim
        query_node_emb = query_node_emb.squeeze(dim=0).unsqueeze(dim=1)  # batch_size, 1, entity_dim
        # query_node_emb = query_node_emb[-1]
        query_node_emb = query_node_emb.view(batch_size, -1, 1)

        #-------------------------Preserved-----------------------------------
        seed_entity_types_emb = self.word_embedding(seed_entity_types)
        _, (seed_entity_types_hidden_emb, _) = self.seed_type_encoder(self.lstm_drop(seed_entity_types_emb),
                                                                      self.init_hidden(1, batch_size,
                                                                                       self.entity_dim))
        seed_entity_types_hidden_emb = seed_entity_types_hidden_emb.view(batch_size, -1, 1)
        question_seed_entity = torch.cat((query_node_emb, seed_entity_types_hidden_emb), 2).view(batch_size, 2, -1)

        _, (question_seed_entity_hidden_emb, _) = self.question_seed_encoder(question_seed_entity,
                                                                             self.init_hidden(1, batch_size,
                                                                                              self.entity_dim))

        question_seed_entity_hidden_emb = question_seed_entity_hidden_emb.view(batch_size, -1, 1)
        # -------------------------Preserved-----------------------------------

        # ----------------------Original----------------------
        # relation_text = relation_text.view(-1, relation_text.shape[3])
        # rel_word_emb = self.word_embedding(relation_text)
        # _, (rel_hidden_emb, _) = self.relation_encoder(self.lstm_drop(rel_word_emb),
        #                                                self.init_hidden(self.num_lstm_layer, relation_text.shape[0],
        #                                                                 self.entity_dim))
        # rel_hidden_emb = rel_hidden_emb[-1]
        # rel_hidden_emb = rel_hidden_emb.view(batch_size, max_rel_paths, -1, max_rel_num)
        # rel_hidden_emb = self.relation_weight(rel_hidden_emb).squeeze(3)
        # ----------------------Original----------------------

        local_rel_emb = self.relation_embedding(local_kb_rel_path_rels)
        local_rel_emb = local_rel_emb.view(-1, local_rel_emb.shape[2], local_rel_emb.shape[3])
        local_rel_hidden_emb, (local_rel_node_emb, _) = self.relation_encoder(self.lstm_drop(local_rel_emb),
                                                                              self.init_hidden(self.num_lstm_layer,
                                                                                               local_rel_emb.shape[0],
                                                                                               self.entity_dim))
        local_rel_node_emb = local_rel_node_emb.squeeze(dim=0).unsqueeze(dim=1)  # batch_size, 1, entity_dim
        # local_rel_node_emb = local_rel_node_emb[-1]
        local_rel_node_emb = local_rel_node_emb.view(batch_size, max_rel_paths, -1)

        # Attention
        div = float(np.sqrt(self.entity_dim))
        local_rel_emb = self.relation_linear(local_rel_node_emb)

        rel_score = local_rel_emb @ query_node_emb
        rel_score = (rel_score / div).squeeze(2) + (1 - local_kb_rel_path_rels_mask) * VERY_NEG_NUMBER

        loss = self.bce_loss(rel_score, answer_dist)

        pred = torch.topk(rel_score, 3, dim=1)[1]
        # pred_max = torch.argmax(rel_score, dim=1)

        # top_relation_emb = local_rel_node_emb[torch.arange(local_rel_node_emb.shape[0]), pred_max]
        # question_seed_entity_hidden_emb = question_seed_entity_hidden_emb.view(batch_size, -1)
        # query_top_relation_emb = torch.cat([question_seed_entity_hidden_emb, top_relation_emb], dim=1)
        # eod_score = self.linear_drop(self.eod_linear1(query_top_relation_emb))
        # eod_score = self.linear_drop(self.eod_linear1_1(self.relu(eod_score)))
        # eod_score = self.eod_linear2(self.relu(eod_score))
        # eod_score = self.eod_linear3(self.relu(eod_score))
        # eod_loss = self.bce_loss(eod_score, local_kb_rel_eods)
        # pred_eod = (self.sigmoid(eod_score) > 0.5) * 1

        return loss, pred, None  # pred, pred_dist

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return (use_cuda(torch.tensor(torch.zeros(num_layer, batch_size, hidden_size))),
                use_cuda(torch.tensor(torch.zeros(num_layer, batch_size, hidden_size))))