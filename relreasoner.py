import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Variable

import numpy as np
import random

from util import use_cuda, read_padded, sparse_bmm
from transformers import *


VERY_NEG_NUMBER = -10000
EOD_TOKEN = 646 # TODO: fix magic number


class QuestionEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, pre_emb_file=None):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(num_embeddings=input_dim + 1, embedding_dim=emb_dim, padding_idx=input_dim)
        if pre_emb_file is not None:
            self.embedding.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(pre_emb_file), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'), requires_grad=False)
            print('load question word emb')

        self.question_text_encoder = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=n_layers, bidirectional=False, dropout=dropout)
        self.seed_type_encoder = nn.LSTM(input_size=emb_dim, hidden_size=hid_dim, num_layers=1, bidirectional=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_question, src_seed):
        batch_size = src_question.shape[0]

        q_embedded = self.dropout(self.embedding(src_question))  # batch_size, max_query_word, word_dim
        q_output, (q_hidden, q_cell) = self.question_text_encoder(q_embedded)

        # q_hidden = q_hidden.squeeze(dim=0).unsqueeze(dim=1)  # batch_size, 1, entity_dim
        # q_hidden = q_hidden[-1]
        # q_hidden = q_hidden.view(batch_size, -1, 1)

        # -------------------------Preserved-----------------------------------
        seed_emb = self.dropout(self.embedding(src_seed))
        _, (seed_hidden, _) = self.seed_type_encoder(seed_emb)
        q_seed_cat = torch.cat((q_hidden, seed_hidden), 0)

        encoder_output, (hidden, cell) = self.question_text_encoder(q_seed_cat, (q_hidden, q_cell))

        return q_output, (hidden, cell)

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return (use_cuda(torch.tensor(torch.zeros(num_layer, batch_size, hidden_size))),
                use_cuda(torch.tensor(torch.zeros(num_layer, batch_size, hidden_size))))


class RelationChainDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, max_len, dropout, pre_emb_file=None):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.max_len = max_len

        self.embedding = nn.Embedding(num_embeddings=output_dim + 1, embedding_dim=emb_dim, padding_idx=output_dim)
        if pre_emb_file is not None:
            self.embedding.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(pre_emb_file), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'), requires_grad=False)
            print('load relation chain emb')

        self.relation_encoder = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.hidden_linear = nn.Linear(hid_dim, hid_dim)
        # self.hidden_linear2 = nn.Linear(hid_dim, hid_dim // 4)
        # self.hidden_linear3 = nn.Linear(hid_dim // 4, hid_dim // 8)
        self.attn = nn.Linear(self.hid_dim * 2, self.max_len)
        self.attn_combine = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_relation, hidden, cell, encoder_output,
                entities, facts, relation2id, reverse_relation2id):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]
        input_relation = input_relation.unsqueeze(0)

        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input_relation))

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[-1]), dim=1)), dim=1)
        attn_weights = attn_weights.unsqueeze(1)
        encoder_output = encoder_output.view(encoder_output.shape[1], encoder_output.shape[0], -1)
        attn_applied = torch.bmm(attn_weights, encoder_output)

        output = torch.cat((attn_applied.squeeze(1), embedded.squeeze(0)), dim=1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        # embedded = [1, batch size, emb dim]
        output, (hidden, cell) = self.relation_encoder(output, (hidden, cell))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        output = self.hidden_linear(output.squeeze(0))
        # output = self.relu(self.hidden_linear2(output))
        # output = self.relu(self.hidden_linear3(output))

        prediction = self.fc_out(output)

        # prediction = [batch size, output dim]
        return prediction, hidden, cell


class RelReasoner(nn.Module):
    def __init__(self, pre_word_emb_file, pre_relation_emb_file, num_relation, num_entity, num_word,
                 num_hop, hidden_dim, word_dim, num_layer, dropout, max_len):
        super(RelReasoner, self).__init__()

        self.num_relation = num_relation
        self.num_hop = num_hop
        self.num_entity = num_entity
        self.num_word = num_word
        self.hidden_dim = hidden_dim
        self.word_dim = word_dim
        self.num_layer = num_layer
        self.max_len = max_len

        self.encoder = QuestionEncoder(num_word, word_dim, hidden_dim, num_layer, dropout, pre_word_emb_file)
        self.decoder = RelationChainDecoder(num_relation, word_dim, hidden_dim, num_layer, max_len, dropout, pre_relation_emb_file)

        def init_weights(m):
            for name, param in m.named_parameters():
                nn.init.uniform_(param.data, -0.08, 0.08)

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)

        # loss
        self.criterion = nn.CrossEntropyLoss()
        self.relu = nn.ReLU()

    def next_hop_entities(self, entities, top1, facts, reverse_relation2id):
        res = []
        for idx in range(entities.shape[0]):
            each_entities = entities[idx]
            each_relation = reverse_relation2id[top1[idx].item()]
            next_entities = set()
            for each_entity in each_entities[:3]:
                if each_entity in facts and each_relation in facts[each_entity]:
                    next_entities.update(list(facts[each_entity][each_relation].keys()))
            next_entities = list(next_entities)
            res.append(next_entities)
        return np.array(res, dtype=np.object)

    def _create_relation_mask(self, entities, output_dim, facts, relation2id, reverse_relation2id):
        mask = np.full((entities.shape[0], output_dim), 0, dtype=np.float32)
        for idx in range(entities.shape[0]):
            each_topic_entities = entities[idx]
            next_relations = set()
            for each_topic_entity in each_topic_entities[:3]:
                if each_topic_entity in facts:
                    next_relations.update(list(facts[each_topic_entity].keys()))
            next_relations_ids = np.array([relation2id[e] for e in next_relations if e in relation2id])
            if next_relations_ids.shape[0] > 0:
                mask[idx][next_relations_ids] = 1
            else:
                mask[idx][:] = 1
        mask = use_cuda(torch.tensor(mask))
        return mask

    def forward(self, batch, teacher_forcing_ratio=0.5, facts=None, relation2id=None, reverse_relation2id=None):
        # query_text are question texts
        # seed_entity_type are seed entities of each question
        # entities are topic entities which could use to filter unnecessary relations (Not used now)
        # targets are the correct relation chain labels the models learn.
        query_text, seed_entity_types, entities, targets = batch

        batch_size = query_text.shape[1]

        # numpy to tensor
        with torch.no_grad():
            query_text = use_cuda(Variable(torch.from_numpy(query_text).type('torch.LongTensor')))
            seed_entity_types = use_cuda(Variable(torch.from_numpy(seed_entity_types).type('torch.LongTensor')))
            targets = use_cuda(Variable(torch.from_numpy(targets).type('torch.LongTensor')))

        encoder_output, (hidden, cell) = self.encoder(query_text, seed_entity_types)

        #tensor to store decoder outputs
        outputs = use_cuda(torch.zeros(self.num_hop + 1, batch_size, self.num_relation))
        preds = use_cuda(torch.zeros(self.num_hop, batch_size))

        # first input to the decoder is the <sos> tokens
        decoder_input = targets[0, :]

        mask = None
        mask = self._create_relation_mask(entities, self.num_relation, facts, relation2id, reverse_relation2id)

        for t in range(1, self.num_hop + 1):
            # t1 = time.time()
            # print('t1', time.time() - t1)
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_output,
                                                entities, facts, relation2id, reverse_relation2id)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # if t == 1:
            #     output = output * mask

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            preds[t - 1] = top1

            # t1 = time.time()
            # entities = self.next_hop_entities(entities, top1, facts, reverse_relation2id)
            # print('t2', time.time() - t1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            decoder_input = targets[t] if teacher_force else top1

        output_dim = outputs.shape[-1]
        outputs = outputs[1:].view(-1, output_dim)
        targets = targets[1:].view(-1)

        loss = self.criterion(outputs, targets)

        return loss, preds, None  # pred, pred_dist

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return (use_cuda(torch.tensor(torch.zeros(num_layer, batch_size, hidden_size))),
                use_cuda(torch.tensor(torch.zeros(num_layer, batch_size, hidden_size))))