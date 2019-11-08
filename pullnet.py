import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from util import use_cuda, read_padded
from torch.sparse import mm as sparse_mm


VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


def sparse_bmm(a, b):
    return torch.stack([sparse_mm(ai, bi) for ai, bi in zip(a, b)])


class PullNet(nn.Module):
    def __init__(self, facts, entity2id, relation2id, pretrained_word_embedding_file, pretrained_entity_emb_file, pretrained_entity_kge_file,
                 pretrained_relation_emb_file, pretrained_relation_kge_file, num_layer, num_relation, num_entity,
                 num_word, entity_dim, word_dim, kge_dim, pagerank_lambda, fact_scale, lstm_dropout, linear_dropout, fact_dropout,
                 use_kb, use_doc, use_inverse_relation):
        """
        num_relation: number of relation including self-connection
        """
        super(PullNet, self).__init__()
        self.facts = facts
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.num_layer = num_layer
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.num_word = num_word
        self.entity_dim = entity_dim
        self.word_dim = word_dim
        self.pagerank_lambda = pagerank_lambda
        self.fact_scale = fact_scale
        self.fact_dropout = fact_dropout
        self.has_entity_kge = False
        self.has_relation_kge = False
        self.use_kb = use_kb
        self.use_doc = use_doc
        self.use_inverse_relation = use_inverse_relation

        # initialize entity embedding
        self.entity_embedding = nn.Embedding(num_embeddings=num_entity + 1, embedding_dim=word_dim,
                                             padding_idx=num_entity, sparse=True)
        if pretrained_entity_emb_file is not None:
            self.entity_embedding.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(pretrained_entity_emb_file), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
            self.entity_embedding.weight.requires_grad = False
        if pretrained_entity_kge_file is not None:
            self.has_entity_kge = True
            self.entity_kge = nn.Embedding(num_embeddings=num_entity + 1, embedding_dim=kge_dim, padding_idx=num_entity)
            self.entity_kge.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(pretrained_entity_kge_file), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
            self.entity_kge.weight.requires_grad = False

        if self.has_entity_kge:
            self.entity_linear = nn.Linear(in_features=word_dim + kge_dim, out_features=entity_dim)
        else:
            self.entity_linear = nn.Linear(in_features=word_dim, out_features=entity_dim)

        # initialize relation embedding
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
        if pretrained_relation_kge_file is not None:
            self.has_relation_kge = True
            self.relation_kge = nn.Embedding(num_embeddings=num_relation + 1, embedding_dim=kge_dim,
                                             padding_idx=num_relation)
            self.relation_kge.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(pretrained_relation_kge_file), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))

        if self.has_relation_kge:
            self.relation_linear = nn.Linear(in_features=2 * word_dim + kge_dim, out_features=entity_dim)
        else:
            self.relation_linear = nn.Linear(in_features=2 * word_dim, out_features=entity_dim)

        # create linear functions
        self.k = 2 + (self.use_kb or self.use_doc)
        for i in range(self.num_layer):
            self.add_module('q2e_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('d2e_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('e2q_linear' + str(i), nn.Linear(in_features=self.k * entity_dim, out_features=entity_dim))
            self.add_module('e2d_linear' + str(i), nn.Linear(in_features=self.k * entity_dim, out_features=entity_dim))
            self.add_module('e2e_linear' + str(i), nn.Linear(in_features=self.k * entity_dim, out_features=entity_dim))

            if self.use_kb:
                self.add_module('kb_head_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
                self.add_module('kb_tail_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
                self.add_module('kb_self_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))

        # initialize text embeddings
        self.word_embedding = nn.Embedding(num_embeddings=num_word + 1, embedding_dim=word_dim, padding_idx=num_word)
        if pretrained_word_embedding_file is not None:
            self.word_embedding.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(pretrained_word_embedding_file), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
            self.word_embedding.weight.requires_grad = False

        # create LSTMs
        self.node_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim, batch_first=True, bidirectional=False)
        self.query_rel_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim, batch_first=True,
                                         bidirectional=False)  # not used
        self.bi_text_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim, batch_first=True,
                                       bidirectional=True)
        self.doc_info_carrier = nn.LSTM(input_size=entity_dim, hidden_size=entity_dim, batch_first=True,
                                        bidirectional=True)
        # dropout
        self.lstm_drop = nn.Dropout(p=lstm_dropout)
        self.linear_drop = nn.Dropout(p=linear_dropout)
        # non-linear activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        # scoring function
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        # loss
        self.log_softmax = nn.LogSoftmax(dim=1)  # pylint: disable=E1123
        self.softmax_d1 = nn.Softmax(dim=1)
        self.kld_loss = nn.KLDivLoss()
        self.bce_loss = nn.BCELoss()
        self.bce_loss_logits = nn.BCEWithLogitsLoss()

    def _prepare_question_subgraph(self, question, h_q, iteration_t):
        entities = question['entities']
        paths = question['path']
        related_entities = set()
        target_entities = set()
        if not paths:
            if 'subgraph' not in question:
                question['subgraph'] = {}
            if not question['subgraph']:
                question['subgraph']['entities'] = question['entities']
                question['subgraph']['tuples'] = []
            return [], related_entities, 1
        for path in paths:
            if len(path) % 2 == 0:
                continue
            if iteration_t < len(path):
                target_entities.update(path[iteration_t * 2])
        if not target_entities:
            if 'subgraph' not in question:
                question['subgraph'] = {}
            if not question['subgraph']:
                question['subgraph']['entities'] = question['entities']
                question['subgraph']['tuples'] = []
            return [], related_entities, 1

        related_relations = set()
        for entity in entities:
            if entity in self.facts:
                related_relations.update(self.facts[entity])

        top_facts = 100
        related_relations = list(sorted(related_relations))
        related_relations_ids = use_cuda(torch.tensor([self.relation2id[rel] for rel in related_relations]))
        related_relation_emb = self.relation_embedding(related_relations_ids)
        relation_scores = torch.sigmoid(self.relation_linear(related_relation_emb) @ (h_q.reshape(h_q.size(0), 1)))
        top_scores = torch.argsort(relation_scores, dim=0)
        top_relations = [related_relations[e.item()] for e in top_scores.cpu()]

        extracted_tuples = set()
        flag = False
        for top_relation in top_relations:
            if flag:
                break
            for entity in entities:
                if flag:
                    break
                if top_relation not in self.facts[entity]:
                    continue
                for obj in self.facts[entity][top_relation]:
                    direction = self.facts[entity][top_relation][obj]
                    if direction == 0:
                        extracted_tuples.add((entity, top_relation, obj))
                    else:
                        extracted_tuples.add((obj, top_relation, entity))
                    if len(extracted_tuples) >= top_facts:
                        flag = True
                        break
        for s, p, o in extracted_tuples:
            related_entities.add(s)
            related_entities.add(o)
        extracted_tuples = list(sorted(extracted_tuples))
        extracted_entities = list(sorted(related_entities))

        if 'subgraph' not in question:
            question['subgraph'] = {}
        if not question['subgraph']:
            question['subgraph']['entities'] = extracted_entities
            question['subgraph']['tuples'] = extracted_tuples
        question['answers_hop_%d' % iteration_t] = list(map(lambda x: {'answer_id': x}, list(sorted(target_entities))))
        return len(related_entities & target_entities) / len(target_entities)

    @staticmethod
    def _add_entity_to_map(entity2id, entities, g2l):
        for entity in entities:
            if isinstance(entity, dict):
                entity_text = entity['text']
            else:
                entity_text = entity
            if entity_text not in entity2id:
                continue
            entity_global_id = entity2id[entity_text]
            if entity_global_id not in g2l:
                g2l[entity2id[entity_text]] = len(g2l)

    def _build_global2local_entity_maps(self, data):
        """Create a map from global entity id to local entity of each sample"""
        global2local_entity_maps = [None] * len(data)
        total_local_entity = 0.0
        next_id = 0
        max_local_entity = 0
        for sample in data:
            g2l = dict()
            self._add_entity_to_map(self.entity2id, sample['entities'], g2l)
            # construct a map from global entity id to local entity id
            if self.use_kb:
                self._add_entity_to_map(self.entity2id, sample['subgraph']['entities'], g2l)
            if self.use_doc:
                for relevant_doc in sample['passages']:
                    if relevant_doc['document_id'] not in self.documents:
                        continue
                    document = self.documents[int(relevant_doc['document_id'])]
                    self._add_entity_to_map(self.entity2id, document['document']['entities'], g2l)
                    if 'title' in document:
                        self._add_entity_to_map(self.entity2id, document['title']['entities'], g2l)

            global2local_entity_maps[next_id] = g2l
            total_local_entity += len(g2l)
            max_local_entity = max(max_local_entity, len(g2l))
            next_id += 1
        print('avg local entity: ', total_local_entity / next_id)
        return global2local_entity_maps, max_local_entity

    def _prepare_data(self, data, iteration_t, global2local_entity_maps, local_entities, kb_adj_mats,
                      kb_fact_rels, q2e_adj_mats, answer_dists):
        next_id = 0
        for sample in data:
            # get a list of local entities
            g2l = global2local_entity_maps[next_id]
            for global_entity, local_entity in g2l.items():
                if local_entity != 0:  # skip question node
                    local_entities[next_id, local_entity] = global_entity

            entity2fact_e, entity2fact_f = [], []
            fact2entity_f, fact2entity_e = [], []

            # relations in local KB
            if self.use_kb:
                for i, tpl in enumerate(sample['subgraph']['tuples']):
                    sbj, rel, obj = tpl
                    if not self.use_inverse_relation:
                        entity2fact_e += [g2l[self.entity2id[sbj]]]
                        entity2fact_f += [i]
                        fact2entity_f += [i]
                        fact2entity_e += [g2l[self.entity2id[obj]]]
                        kb_fact_rels[next_id, i] = self.relation2id[rel]
                    else:
                        entity2fact_e += [g2l[self.entity2id[sbj]], g2l[self.entity2id[obj]]]
                        entity2fact_f += [2 * i, 2 * i + 1]
                        fact2entity_f += [2 * i, 2 * i + 1]
                        fact2entity_e += [g2l[self.entity2id[obj]], g2l[self.entity2id[sbj]]]
                        kb_fact_rels[next_id, 2 * i] = self.relation2id[rel]
                        kb_fact_rels[next_id, 2 * i + 1] = self.relation2id[rel] + len(self.relation2id)

            # build connection between question and entities in it
            for j, entity in enumerate(sample['entities']):
                if entity not in self.entity2id:
                    continue
                q2e_adj_mats[next_id, g2l[self.entity2id[entity]], 0] = 1.0

            # construct distribution for answers
            for answer in sample['answers_hop_%d' % iteration_t]:
                keyword = 'answer_id'
                if answer[keyword] in self.entity2id and self.entity2id[answer[keyword]] in g2l:
                    answer_dists[next_id, g2l[self.entity2id[answer[keyword]]]] = 1.0

            kb_adj_mats[next_id] = (np.array(entity2fact_f, dtype=int), np.array(entity2fact_e, dtype=int),
                                         np.array([1.0] * len(entity2fact_f))), (
                                            np.array(fact2entity_e, dtype=int), np.array(fact2entity_f, dtype=int),
                                            np.array([1.0] * len(fact2entity_e)))
            next_id += 1

    def _build_kb_adj_mat(self, kb_adj_mats, fact_dropout):
        """Create sparse matrix representation for batched data"""
        mats0_batch = np.array([], dtype=int)
        mats0_0 = np.array([], dtype=int)
        mats0_1 = np.array([], dtype=int)
        vals0 = np.array([], dtype=float)

        mats1_batch = np.array([], dtype=int)
        mats1_0 = np.array([], dtype=int)
        mats1_1 = np.array([], dtype=int)
        vals1 = np.array([], dtype=float)

        for sample_id in range(len(kb_adj_mats)):
            (mat0_0, mat0_1, val0), (mat1_0, mat1_1, val1) = kb_adj_mats[sample_id]
            assert len(val0) == len(val1)
            num_fact = len(val0)
            num_keep_fact = int(np.floor(num_fact * (1 - fact_dropout)))
            mask_index = np.random.permutation(num_fact)[: num_keep_fact]
            # mat0
            mats0_batch = np.append(mats0_batch, np.full(len(mask_index), sample_id, dtype=int))
            mats0_0 = np.append(mats0_0, mat0_0[mask_index])
            mats0_1 = np.append(mats0_1, mat0_1[mask_index])
            vals0 = np.append(vals0, val0[mask_index])
            # mat1
            mats1_batch = np.append(mats1_batch, np.full(len(mask_index), sample_id, dtype=int))
            mats1_0 = np.append(mats1_0, mat1_0[mask_index])
            mats1_1 = np.append(mats1_1, mat1_1[mask_index])
            vals1 = np.append(vals1, val1[mask_index])

        return (mats0_batch, mats0_0, mats0_1, vals0), (mats1_batch, mats1_0, mats1_1, vals1)

    def pull_facts(self, h_qs, question_data, iteration_t):
        avg_recall = 0
        max_facts = 0
        for idx in range(len(question_data)):
            q = question_data[idx]
            h_q = h_qs[idx]
            recall = self._prepare_question_subgraph(q, h_q[-1], iteration_t)
            max_facts = max(max_facts, 2 * len(q['subgraph']['tuples']))
            avg_recall += recall
        print('pull_facts recall:', avg_recall / len(question_data))

        global2local_entity_maps, max_local_entity = self._build_global2local_entity_maps(question_data)

        num_data = len(question_data)
        local_entities = np.full((num_data, max_local_entity), len(self.entity2id), dtype=int)
        kb_adj_mats = np.empty(num_data, dtype=object)
        kb_fact_rels = np.full((num_data, max_facts), self.num_relation, dtype=int)
        q2e_adj_mats = np.zeros((num_data, max_local_entity, 1), dtype=float)
        answer_dists = np.zeros((num_data, max_local_entity), dtype=float)

        self._prepare_data(question_data, iteration_t, global2local_entity_maps, local_entities, kb_adj_mats, kb_fact_rels, q2e_adj_mats, answer_dists)
        return local_entities, q2e_adj_mats, self._build_kb_adj_mat(kb_adj_mats, fact_dropout=self.fact_dropout), kb_fact_rels, None, None, answer_dists

    def forward(self, batch):
        """
        :local_entity: global_id of each entity                     (batch_size, max_local_entity)
        :q2e_adj_mat: adjacency matrices (dense)                    (batch_size, max_local_entity, 1)
        :kb_adj_mat: adjacency matrices (sparse)                    (batch_size, max_fact, max_local_entity), (batch_size, max_local_entity, max_fact)
        :kb_fact_rel:                                               (batch_size, max_fact)
        :query_text: a list of words in the query                   (batch_size, max_query_word)
        :document_text:                                             (batch_size, max_relevant_doc, max_document_word)
        :entity_pos: sparse entity_pos_mat                          (batch_size, max_local_entity, max_relevant_doc * max_document_word)
        :answer_dist: an distribution over local_entity             (batch_size, max_local_entity)
        """
        query_text, question_data = batch
        with torch.no_grad():
            query_text = use_cuda(Variable(torch.from_numpy(query_text).type('torch.LongTensor')))
            query_mask = use_cuda((query_text != self.num_word).type('torch.FloatTensor'))

        # encode query
        batch_size = batch[0].shape[0]
        query_word_emb = self.word_embedding(query_text)  # batch_size, max_query_word, word_dim
        query_hidden_emb, (query_node_emb, _) = self.node_encoder(
            self.lstm_drop(query_word_emb), self.init_hidden(1, batch_size, self.entity_dim))  # 1, batch_size, entity_dim
        query_node_emb = query_node_emb.squeeze(dim=0).unsqueeze(dim=1)

        local_entity, q2e_adj_mat, kb_adj_mat, kb_fact_rel, document_text, entity_pos, answer_dist = self.pull_facts(query_hidden_emb, question_data, 1)

        batch_size, max_local_entity = local_entity.shape
        _, max_relevant_doc, max_document_word = document_text.shape if self.use_doc else None, None, None
        _, max_fact = kb_fact_rel.shape

        # numpy to tensor
        with torch.no_grad():
            local_entity = use_cuda(Variable(torch.from_numpy(local_entity).type('torch.LongTensor')))
            if self.use_kb:
                kb_fact_rel = use_cuda(Variable(torch.from_numpy(kb_fact_rel).type('torch.LongTensor')))
            if self.use_doc:
                document_text = use_cuda(Variable(torch.from_numpy(document_text).type('torch.LongTensor')))
            answer_dist = use_cuda(
                Variable(torch.from_numpy(answer_dist).type('torch.FloatTensor')))
        local_entity_mask = use_cuda((local_entity != self.num_entity).type('torch.FloatTensor'))
        if self.use_doc:
            document_mask = use_cuda((document_text != self.num_word).type('torch.FloatTensor'))
        # normalized adj matrix
        pagerank_f = use_cuda(Variable(torch.from_numpy(q2e_adj_mat).type('torch.FloatTensor')).squeeze(
            dim=2))  # batch_size, max_local_entity
        pagerank_f.requires_grad = True
        with torch.no_grad():
            pagerank_d = use_cuda(
                Variable(torch.from_numpy(q2e_adj_mat).type('torch.FloatTensor'))).squeeze(
                dim=2)  # batch_size, max_local_entity
            q2e_adj_mat = use_cuda(
                Variable(torch.from_numpy(q2e_adj_mat).type('torch.FloatTensor')))  # batch_size, max_local_entity, 1
        assert pagerank_f.requires_grad == True
        assert pagerank_d.requires_grad == False

        query_rel_emb = query_node_emb  # batch_size, 1, entity_dim

        if self.use_kb:
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
            if self.has_relation_kge:
                local_fact_emb = torch.cat((local_fact_emb, self.relation_kge(kb_fact_rel)),
                                           dim=2)  # batch_size, max_fact, 2 * word_dim + kge_dim
            local_fact_emb = self.relation_linear(local_fact_emb)  # batch_size, max_fact, entity_dim

            # attention fact2question
            div = float(np.sqrt(self.entity_dim))
            fact2query_sim = torch.bmm(query_hidden_emb,
                                       local_fact_emb.transpose(1, 2)) / div  # batch_size, max_query_word, max_fact
            fact2query_sim = self.softmax_d1(fact2query_sim + (
                        1 - query_mask.unsqueeze(dim=2)) * VERY_NEG_NUMBER)  # batch_size, max_query_word, max_fact
            fact2query_att = torch.sum(fact2query_sim.unsqueeze(dim=3) * query_hidden_emb.unsqueeze(dim=2),
                                       dim=1)  # batch_size, max_fact, entity_dim
            W = torch.sum(fact2query_att * local_fact_emb, dim=2) / div  # batch_size, max_fact
            W_max = torch.max(W, dim=1, keepdim=True)[0]  # batch_size, 1
            W_tilde = torch.exp(W - W_max)  # batch_size, max_fact
            e2f_softmax = sparse_bmm(entity2fact_mat.transpose(1, 2), W_tilde.unsqueeze(dim=2)).squeeze(
                dim=2)  # batch_size, max_local_entity
            e2f_softmax = torch.clamp(e2f_softmax, min=VERY_SMALL_NUMBER)
            with torch.no_grad():
                e2f_out_dim = use_cuda(
                    Variable(torch.sum(entity2fact_mat.to_dense(), dim=1)))  # batch_size, max_local_entity

        # build entity_pos matrix
        if self.use_doc:
            entity_pos_dim_batch, entity_pos_dim_entity, entity_pos_dim_doc_by_word, entity_pos_value = entity_pos
            entity_pos_index = torch.LongTensor(
                [entity_pos_dim_batch, entity_pos_dim_entity, entity_pos_dim_doc_by_word])
            entity_pos_val = torch.FloatTensor(entity_pos_value)
            entity_pos_mat = use_cuda(torch.sparse.FloatTensor(entity_pos_index, entity_pos_val, torch.Size(
                [batch_size, max_local_entity,
                 max_relevant_doc * max_document_word])))  # batch_size, max_local_entity, max_relevant_doc * max_document_word
            d2e_adj_mat = torch.sum(
                entity_pos_mat.to_dense().view(batch_size, max_local_entity, max_relevant_doc, max_document_word),
                dim=3)  # batch_size, max_local_entity, max_relevant_doc
            with torch.no_grad():
                d2e_adj_mat = use_cuda(Variable(d2e_adj_mat))
            e2d_out_dim = torch.sum(torch.sum(
                entity_pos_mat.to_dense().view(batch_size, max_local_entity, max_relevant_doc, max_document_word),
                dim=3), dim=2, keepdim=True)  # batch_size, max_local_entity, 1
            with torch.no_grad():
                e2d_out_dim = use_cuda(Variable(e2d_out_dim))
            e2d_out_dim = torch.clamp(e2d_out_dim, min=VERY_SMALL_NUMBER)

            d2e_out_dim = torch.sum(torch.sum(
                entity_pos_mat.to_dense().view(batch_size, max_local_entity, max_relevant_doc, max_document_word),
                dim=3), dim=1, keepdim=True)  # batch_size, 1, max_relevant_doc
            with torch.no_grad():
                d2e_out_dim = use_cuda(Variable(d2e_out_dim))
            d2e_out_dim = torch.clamp(d2e_out_dim, min=VERY_SMALL_NUMBER).transpose(1, 2)

            # encode document
            document_textual_emb = self.word_embedding(document_text.view(batch_size * max_relevant_doc,
                                                                          max_document_word))  # batch_size * max_relevant_doc, max_document_word, entity_dim
            document_textual_emb, (document_node_emb, _) = read_padded(self.bi_text_encoder,
                                                                       self.lstm_drop(document_textual_emb),
                                                                       document_mask.view(-1,
                                                                                          max_document_word))  # batch_size * max_relevant_doc, max_document_word, entity_dim
            document_textual_emb = document_textual_emb[:, :, 0: self.entity_dim] + document_textual_emb[:, :,
                                                                                    self.entity_dim:]
            document_textual_emb = document_textual_emb.contiguous().view(batch_size, max_relevant_doc,
                                                                          max_document_word, self.entity_dim)
            document_node_emb = (document_node_emb[0, :, :] + document_node_emb[1, :, :]).view(batch_size,
                                                                                               max_relevant_doc,
                                                                                               self.entity_dim)  # batch_size, max_relevant_doc, entity_dim

        # load entity embedding
        local_entity_emb = self.entity_embedding(local_entity)  # batch_size, max_local_entity, word_dim
        if self.has_entity_kge:
            local_entity_emb = torch.cat((local_entity_emb, self.entity_kge(local_entity)),
                                         dim=2)  # batch_size, max_local_entity, word_dim + kge_dim
        if self.word_dim != self.entity_dim:
            local_entity_emb = self.entity_linear(local_entity_emb)  # batch_size, max_local_entity, entity_dim

        # label propagation on entities
        for i in range(self.num_layer):
            # get linear transformation functions for each layer
            q2e_linear = getattr(self, 'q2e_linear' + str(i))
            d2e_linear = getattr(self, 'd2e_linear' + str(i))
            e2q_linear = getattr(self, 'e2q_linear' + str(i))
            e2d_linear = getattr(self, 'e2d_linear' + str(i))
            e2e_linear = getattr(self, 'e2e_linear' + str(i))
            if self.use_kb:
                kb_self_linear = getattr(self, 'kb_self_linear' + str(i))
                kb_head_linear = getattr(self, 'kb_head_linear' + str(i))
                kb_tail_linear = getattr(self, 'kb_tail_linear' + str(i))

            # start propagation
            next_local_entity_emb = local_entity_emb

            # STEP 1: propagate from question, documents, and facts to entities
            # question -> entity
            q2e_emb = q2e_linear(self.linear_drop(query_node_emb)).expand(batch_size, max_local_entity,
                                                                          self.entity_dim)  # batch_size, max_local_entity, entity_dim
            next_local_entity_emb = torch.cat((next_local_entity_emb, q2e_emb),
                                              dim=2)  # batch_size, max_local_entity, entity_dim * 2

            # document -> entity
            if self.use_doc:
                pagerank_e2d = sparse_bmm(entity_pos_mat.transpose(1, 2), pagerank_d.unsqueeze(
                    dim=2) / e2d_out_dim)  # batch_size, max_relevant_doc * max_document_word, 1
                pagerank_e2d = pagerank_e2d.view(batch_size, max_relevant_doc, max_document_word)
                pagerank_e2d = torch.sum(pagerank_e2d, dim=2)  # batch_size, max_relevant_doc
                pagerank_e2d = pagerank_e2d / torch.clamp(torch.sum(pagerank_e2d, dim=1, keepdim=True),
                                                          min=VERY_SMALL_NUMBER)  # batch_size, max_relevant_doc
                pagerank_e2d = pagerank_e2d.unsqueeze(dim=2).expand(batch_size, max_relevant_doc,
                                                                    max_document_word)  # batch_size, max_relevant_doc, max_document_word
                pagerank_e2d = pagerank_e2d.contiguous().view(batch_size,
                                                              max_relevant_doc * max_document_word)  # batch_size, max_relevant_doc * max_document_word
                pagerank_d2e = sparse_bmm(entity_pos_mat,
                                          pagerank_e2d.unsqueeze(dim=2))  # batch_size, max_local_entity, 1
                pagerank_d2e = pagerank_d2e.squeeze(dim=2)  # batch_size, max_local_entity
                pagerank_d2e = pagerank_d2e / torch.clamp(torch.sum(pagerank_d2e, dim=1, keepdim=True),
                                                          min=VERY_SMALL_NUMBER)
                pagerank_d = self.pagerank_lambda * pagerank_d2e + (1 - self.pagerank_lambda) * pagerank_d

                d2e_emb = sparse_bmm(entity_pos_mat, d2e_linear(
                    document_textual_emb.view(batch_size, max_relevant_doc * max_document_word, self.entity_dim)))
                d2e_emb = d2e_emb * pagerank_d.unsqueeze(dim=2)  # batch_size, max_local_entity, entity_dim

            # fact -> entity
            if self.use_kb:
                e2f_emb = self.relu(kb_self_linear(local_fact_emb) + sparse_bmm(entity2fact_mat, kb_head_linear(
                    self.linear_drop(local_entity_emb))))  # batch_size, max_fact, entity_dim
                e2f_softmax_normalized = W_tilde.unsqueeze(dim=2) * sparse_bmm(entity2fact_mat,
                                                                               (pagerank_f / e2f_softmax).unsqueeze(
                                                                                   dim=2))  # batch_size, max_fact, 1
                e2f_emb = e2f_emb * e2f_softmax_normalized  # batch_size, max_fact, entity_dim
                f2e_emb = self.relu(kb_self_linear(local_entity_emb) + sparse_bmm(fact2entity_mat, kb_tail_linear(
                    self.linear_drop(e2f_emb))))

                pagerank_f = self.pagerank_lambda * sparse_bmm(fact2entity_mat, e2f_softmax_normalized).squeeze(
                    dim=2) + (1 - self.pagerank_lambda) * pagerank_f  # batch_size, max_local_entity

            # STEP 2: combine embeddings from fact and documents
            if self.use_doc and self.use_kb:
                next_local_entity_emb = torch.cat((next_local_entity_emb, self.fact_scale * f2e_emb + d2e_emb),
                                                  dim=2)  # batch_size, max_local_entity, entity_dim * 3
            elif self.use_doc:
                next_local_entity_emb = torch.cat((next_local_entity_emb, d2e_emb),
                                                  dim=2)  # batch_size, max_local_entity, entity_dim * 3
            elif self.use_kb:
                next_local_entity_emb = torch.cat((next_local_entity_emb, self.fact_scale * f2e_emb),
                                                  dim=2)  # batch_size, max_local_entity, entity_dim * 3
            else:
                assert False, 'using neither kb nor doc ???'

            # STEP 3: propagate from entities to update question, documents, and facts
            # entity -> document
            if self.use_doc:
                e2d_emb = torch.bmm(d2e_adj_mat.transpose(1, 2), e2d_linear(
                    self.linear_drop(next_local_entity_emb / e2d_out_dim)))  # batch_size, max_relevant_doc, entity_dim
                e2d_emb = sparse_bmm(entity_pos_mat.transpose(1, 2), e2d_linear(self.linear_drop(
                    next_local_entity_emb)))  # batch_size, max_relevant_doc * max_document_word, entity_dim
                e2d_emb = e2d_emb.view(batch_size, max_relevant_doc, max_document_word,
                                       self.entity_dim)  # batch_size, max_relevant_doc, max_document_word, entity_dim
                document_textual_emb = document_textual_emb + e2d_emb  # batch_size, max_relevant_doc, max_document_word, entity_dim
                document_textual_emb = document_textual_emb.view(-1, max_document_word, self.entity_dim)
                document_textual_emb, _ = read_padded(self.doc_info_carrier, self.lstm_drop(document_textual_emb),
                                                      document_mask.view(-1,
                                                                         max_document_word))  # batch_size * max_relevant_doc, max_document_word, entity_dim
                document_textual_emb = document_textual_emb[:, :, 0: self.entity_dim] + document_textual_emb[:, :,
                                                                                        self.entity_dim:]
                document_textual_emb = document_textual_emb.contiguous().view(batch_size, max_relevant_doc,
                                                                              max_document_word, self.entity_dim)

            # entity -> query
            query_node_emb = torch.bmm(pagerank_f.unsqueeze(dim=1), e2q_linear(self.linear_drop(next_local_entity_emb)))

            # update entity
            local_entity_emb = self.relu(
                e2e_linear(self.linear_drop(next_local_entity_emb)))  # batch_size, max_local_entity, entity_dim

        # calculate loss and make prediction
        score = self.score_func(self.linear_drop(local_entity_emb)).squeeze(dim=2)  # batch_size, max_local_entity
        loss = self.bce_loss_logits(score, answer_dist)

        score = score + (1 - local_entity_mask) * VERY_NEG_NUMBER
        pred_dist = self.sigmoid(score) * local_entity_mask
        pred = torch.topk(score, 7, dim=1)[1]

        return loss, pred, pred_dist

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return (use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))),
                use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))))
