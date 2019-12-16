import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from util import use_cuda, read_padded, sparse_bmm


VERY_NEG_NUMBER = -100000000000


class FactsPullNet(nn.Module):
    def __init__(self, pretrained_word_embedding_file, pretrained_relation_emb_file, num_relation, num_entity, num_word, entity_dim, word_dim, lstm_dropout, use_inverse_relation):
        super(FactsPullNet, self).__init__()

        self.num_relation = num_relation
        self.num_entity = num_entity
        self.num_word = num_word
        self.entity_dim = entity_dim
        self.word_dim = word_dim

        self.relation_embedding = nn.Embedding(num_embeddings=num_relation + 1, embedding_dim=2 * word_dim,
                                               padding_idx=num_relation)
        if pretrained_relation_emb_file is not None:
            if use_inverse_relation:
                self.relation_embedding.weight = nn.Parameter(
                    torch.from_numpy(np.pad(
                        np.concatenate((np.load(pretrained_relation_emb_file), np.load(pretrained_relation_emb_file)),
                                       axis=0), ((0, 1), (0, 0)), 'constant')).type('torch.FloatTensor'))
            else:
                self.relation_embedding.weight = nn.Parameter(
                    torch.from_numpy(np.pad(np.load(pretrained_relation_emb_file), ((0, 1), (0, 0)), 'constant')).type(
                        'torch.FloatTensor'))

        self.word_embedding = nn.Embedding(num_embeddings=num_word + 1, embedding_dim=word_dim, padding_idx=num_word)
        if pretrained_word_embedding_file is not None:
            self.word_embedding.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(pretrained_word_embedding_file), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
            self.word_embedding.weight.requires_grad = False

        self.relation_linear = nn.Linear(in_features=2 * word_dim, out_features=entity_dim)

        self.node_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim, batch_first=True, bidirectional=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax_d1 = nn.Softmax(dim=1)
        # dropout
        self.lstm_drop = nn.Dropout(p=lstm_dropout)
        # loss
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        local_entity, q2e_adj_mat, kb_adj_mat, kb_fact_rel, query_text, answer_dist = batch

        batch_size, max_local_entity = local_entity.shape
        _, max_fact = kb_fact_rel.shape

        # numpy to tensor
        with torch.no_grad():
            local_entity = use_cuda(Variable(torch.from_numpy(local_entity).type('torch.LongTensor')))
            query_text = use_cuda(Variable(torch.from_numpy(query_text).type('torch.LongTensor')))
            query_mask = use_cuda((query_text != self.num_word).type('torch.FloatTensor'))
            kb_fact_rel = use_cuda(Variable(torch.from_numpy(kb_fact_rel).type('torch.LongTensor')))
            answer_dist = use_cuda(
                Variable(torch.from_numpy(answer_dist).type('torch.FloatTensor')))
        local_entity_mask = use_cuda((local_entity != self.num_entity).type('torch.FloatTensor'))
        # encode query
        query_word_emb = self.word_embedding(query_text)  # batch_size, max_query_word, word_dim
        query_hidden_emb, (query_node_emb, _) = self.node_encoder(self.lstm_drop(query_word_emb),
                                                                  self.init_hidden(1, batch_size,
                                                                                   self.entity_dim))  # 1, batch_size, entity_dim
        query_node_emb = query_node_emb.squeeze(dim=0).unsqueeze(dim=1)  # batch_size, 1, entity_dim
        query_node_emb = query_node_emb.transpose(1, 2)

        # build kb_adj_matrix from sparse matrix
        (e2f_batch, e2f_f, e2f_e, e2f_val), (f2e_batch, f2e_e, f2e_f, f2e_val) = kb_adj_mat
        entity2fact_index = torch.LongTensor([e2f_batch, e2f_f, e2f_e])
        entity2fact_val = torch.FloatTensor(e2f_val)
        entity2fact_mat = use_cuda(torch.sparse.FloatTensor(entity2fact_index, entity2fact_val, torch.Size(
            [batch_size, max_fact, max_local_entity])))  # batch_size, max_fact, max_local_entity

        fact2entity_index = torch.LongTensor([f2e_batch, f2e_e, f2e_f])
        fact2entity_val = torch.FloatTensor(f2e_val)
        fact2entity_mat = use_cuda(torch.sparse.FloatTensor(fact2entity_index, fact2entity_val,
                                                            torch.Size([batch_size, max_local_entity, max_fact])))

        # load fact embedding
        local_fact_emb = self.relation_embedding(kb_fact_rel)  # batch_size, max_fact, 2 * word_dim
        local_fact_emb = self.relation_linear(local_fact_emb)

        # calculating fact score
        div = float(np.sqrt(self.entity_dim))

        # fact2query_sim = torch.bmm(query_hidden_emb,
        #                            local_fact_emb.transpose(1, 2)) / div  # batch_size, max_query_word, max_fact
        # fact2query_sim = self.softmax_d1(fact2query_sim + (
        #             1 - query_mask.unsqueeze(dim=2)) * VERY_NEG_NUMBER)  # batch_size, max_query_word, max_fact
        # fact2query_att = torch.sum(fact2query_sim.unsqueeze(dim=3) * query_hidden_emb.unsqueeze(dim=2),
        #                            dim=1)  # batch_size, max_fact, entity_dim
        # local_fact_att_emb = fact2query_att * local_fact_emb

        fact_score = (local_fact_emb @ query_node_emb / div).squeeze(2)
        # entity_score = sparse_bmm(fact2entity_mat, local_fact_att_emb).squeeze(2) / float(np.sqrt(kb_fact_rel.shape[1]))

        loss = self.bce_loss(fact_score, answer_dist)

        # entity_score = entity_score + (1 - local_entity_mask) * VERY_NEG_NUMBER
        # pred_dist = self.sigmoid(entity_score)* local_entity_mask
        pred = torch.topk(fact_score, 50, dim=1)[1]

        return loss, pred, None#pred, pred_dist

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return (use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))),
                use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))))