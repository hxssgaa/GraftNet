import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from preprocessing.preprocess_metaqa import clean_text
from util import use_cuda, read_padded, sparse_bmm
from transformers import *


VERY_NEG_NUMBER = -100000000000


class RelChainPredictor(nn.Module):
    def __init__(self, facts, relation2id, word2id, pretrained_word_embedding_file, pretrained_relation_emb_file, num_relation, num_entity, num_word, num_hop,
                 entity_dim, word_dim, is_train=False):
        super(RelChainPredictor, self).__init__()

        self.num_relation = num_relation
        self.num_entity = num_entity
        self.num_word = num_word
        self.entity_dim = entity_dim
        self.word_dim = word_dim
        self.facts = facts
        self.relation2id = relation2id
        self.reverse_relation2id = {v: k for k, v in relation2id.items()}
        self.word2id = word2id
        self.num_lstm_layer = 3
        self.num_hop = num_hop
        self.max_local_path_rel = 300
        self.max_query_word = 20
        self.max_seed_entities = 10
        self.is_train = is_train
        self.cache_relation = dict()

        self.relation_embedding = nn.Embedding(num_embeddings=num_relation + 1, embedding_dim=word_dim,
                                               padding_idx=num_relation)
        if pretrained_relation_emb_file is not None:
            self.relation_embedding.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(pretrained_relation_emb_file), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
            self.relation_embedding.weight.requires_grad = False
            print('loaded relation_emb')

        self.word_embedding = nn.Embedding(num_embeddings=num_word + 1, embedding_dim=word_dim, padding_idx=num_word)
        if pretrained_word_embedding_file is not None:
            self.word_embedding.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(pretrained_word_embedding_file), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
            self.word_embedding.weight.requires_grad = False
            print('load word emb')

        self.relation_linear = nn.Linear(in_features=word_dim, out_features=entity_dim)
        # self.hidden1 = nn.Linear(in_features=word_dim, out_features=entity_dim)
        self.relation_weight = nn.Linear(in_features=3, out_features=1)

        self.node_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim, num_layers=self.num_lstm_layer, batch_first=True, bidirectional=False)
        self.relation_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim, num_layers=self.num_lstm_layer, batch_first=True, bidirectional=False)
        self.sigmoid = nn.Sigmoid()
        # self.batch_norm = use_cuda(torch.nn.BatchNorm1d(word_dim if num_hop == 2 else 300))
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

    def _prepare_data(self, batch, num_hop):
        raw_data, query_text, relation_text, local_kb_rel_path_rels, answer_dist = batch

        for idx_q, sample in enumerate(raw_data):
            if num_hop > len(sample['rel_cands_multi']):
                target_rels = ['EOD']
            else:
                target_rels = sample['rel_cands_multi'][num_hop - 1]
            target_rel_ids = set()
            for e in target_rels:
                target_rel_ids.add(self.relation2id[e])

            seed_entities = sample['entities'] if num_hop == 1 else sample['pred_entities%d' % (num_hop - 1)]
            if num_hop > 1 and self.is_train and 'EOD' not in target_rels:
                gt_entities = set([p[(num_hop - 1) * 2] for p in sample['ground_truth_path'] if (num_hop - 1) * 2 < len(p)])
                gt_entities.update(seed_entities)
                seed_entities = list(gt_entities)[:self.max_seed_entities]
            sample['seed_entities'] = seed_entities
            cand_rels = set()
            for e in seed_entities:
                cand_rels.update(list(self.facts[e].keys()))
            if num_hop > 1:
                cand_rels.add('EOD')
            cand_rels = list(cand_rels)
            # filtering invalid cands
            cand_rels = [e for e in cand_rels if (e not in (
                'Equals', 'GreaterThan', 'GreaterThanOrEqual', 'LessThan', 'LessThanOrEqual', 'NotEquals',
                'rdf-schema#domain', 'rdf-schema#range') and not e.startswith('freebase.'))]
            sample['cand_rels%d' % num_hop] = cand_rels
            sample['target_rels%d' % num_hop] = target_rels

            for i in range(min(len(cand_rels), self.max_local_path_rel)):
                # rel = cand_rels[i]
                # rel_spt = rel.split('.')
                # if len(rel_spt) > 3:
                #     rel_spt = rel_spt[-3:]
                # for j in range(len(rel_spt)):
                #     if j - 3 < -len(rel_spt):
                #         rel = '__unk__'
                #     else:
                #         rel = rel_spt[j-3]
                #     if rel not in self.cache_relation:
                #         self.cache_relation[rel] = clean_text(rel, filter_dot=True)
                #     rel_word_spt = self.cache_relation[rel]
                #     for k, word in enumerate(rel_word_spt):
                #         if k < self.max_query_word:
                #             if word in self.word2id:
                #                 relation_text[idx_q, i, j, k] = self.word2id[word]
                #             else:
                #                 relation_text[idx_q, i, j, k] = self.word2id['__unk__']
                local_kb_rel_path_rels[idx_q][i] = self.relation2id[cand_rels[i]]
                if local_kb_rel_path_rels[idx_q][i] in target_rel_ids:
                    answer_dist[idx_q][i] = 1.0

        return raw_data, query_text, relation_text, local_kb_rel_path_rels, answer_dist


    def forward(self, batch):
        raw_data, query_text, relation_text, local_kb_rel_path_rels, answer_dist = batch

        batch_size, max_rel_paths = local_kb_rel_path_rels.shape

        with torch.no_grad():
            query_text = use_cuda(Variable(torch.from_numpy(query_text).type('torch.LongTensor')))
            query_mask = use_cuda((query_text != self.num_word).type('torch.FloatTensor'))

        query_word_emb = self.word_embedding(query_text)  # batch_size, max_query_word, word_dim
        query_hidden_emb, (query_node_emb, _) = self.node_encoder(self.lstm_drop(query_word_emb),
                                                                  self.init_hidden(self.num_lstm_layer, batch_size,
                                                                                   self.entity_dim))  # 1, batch_size, entity_dim
        query_node_emb = query_node_emb.squeeze(dim=0).unsqueeze(dim=1)  # batch_size, 1, entity_dim
        query_node_emb = query_node_emb[-1]
        query_node_emb = query_node_emb.view(batch_size, -1, 1)
        # max_rel_num = relation_text.shape[2]

        # rel_hidden = self.init_hidden(self.num_lstm_layer, relation_text.shape[0] * relation_text.shape[1] * relation_text.shape[2], self.entity_dim)

        total_loss = None
        preds = []
        answer_dists = []
        for t in range(1, 4):

            batch = self._prepare_data(batch, t)
            raw_data, query_text, relation_text, local_kb_rel_path_rels_raw, answer_dist = batch
            answer_dists.append(answer_dist)

            # numpy to tensor
            with torch.no_grad():
                # relation_text = use_cuda(Variable(torch.from_numpy(relation_text).type('torch.LongTensor')))
                # relation_mask = use_cuda((relation_text != self.num_word).type('torch.FloatTensor'))
                local_kb_rel_path_rels = use_cuda(
                    Variable(torch.from_numpy(local_kb_rel_path_rels_raw).type('torch.LongTensor')))
                local_kb_rel_path_rels_mask = use_cuda((local_kb_rel_path_rels != self.num_relation).type('torch.FloatTensor'))
                answer_dist = use_cuda(
                    Variable(torch.from_numpy(answer_dist).type('torch.FloatTensor')))

            # relation embedding
            div = float(np.sqrt(self.entity_dim))

            local_kb_rel_path_rels = self.relation_embedding(local_kb_rel_path_rels)

            local_rel_emb = self.relation_linear(local_kb_rel_path_rels)

            # calculating score

            rel_score = local_rel_emb @ query_node_emb
            rel_score = (rel_score / div).squeeze(2) + (1 - local_kb_rel_path_rels_mask) * VERY_NEG_NUMBER
            loss = self.bce_loss(rel_score, answer_dist)
            if not total_loss:
                total_loss = loss
            else:
                total_loss += loss

            pred = torch.topk(rel_score, 3, dim=1)[1]

            for idx in range(pred.shape[0]):
                sample = raw_data[idx]
                each_pred = list(pred[idx].cpu().numpy())
                each_pred_rel_ids = [local_kb_rel_path_rels_raw[idx][e] for e in each_pred]
                each_pred_rels = [self.reverse_relation2id[e] for e in each_pred_rel_ids if e in self.reverse_relation2id]
                sample['pred_rels%d' % t] = each_pred_rels
                each_pred_entities = set()
                for entity in sample['seed_entities']:
                    for rel in each_pred_rels:
                        if rel in self.facts[entity]:
                            each_pred_entities.update(list(self.facts[entity][rel].keys()))
                each_pred_entities = list(each_pred_entities)[:10]
                sample['pred_entities%d' % t] = each_pred_entities

            preds.append(pred.cpu().numpy())

        return total_loss, preds[1], answer_dists[1]  # pred, pred_dist

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return (use_cuda(torch.tensor(torch.zeros(num_layer, batch_size, hidden_size))),
                use_cuda(torch.tensor(torch.zeros(num_layer, batch_size, hidden_size))))