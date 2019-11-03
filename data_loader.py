import json
import numpy as np

from scipy.sparse import coo_matrix, csr_matrix

from preprocessing.process_complex_webq import clean_text
from util import load_dict, load_json
from tqdm import tqdm
from torch.utils.data import Dataset
np.random.seed(1024)


class GraftNetDataSet(Dataset):
    def __init__(self, data_file, documents, document_entity_indices, document_texts, word2id, relation2id, entity2id,
                 max_query_word, max_document_word, use_kb, use_doc, use_inverse_relation, fact_dropout):
        self.use_kb = use_kb
        self.use_doc = use_doc
        self.use_inverse_relation = use_inverse_relation
        self.max_local_entity = 0
        self.max_relevant_doc = 0
        self.max_facts = 0
        self.max_query_word = max_query_word
        self.max_document_word = max_document_word
        if self.use_kb:
            if self.use_inverse_relation:
                self.num_kb_relation = 2 * len(relation2id)
            else:
                self.num_kb_relation = len(relation2id)
        else:
            self.num_kb_relation = 0

        print('loading data from', data_file)
        self.data = []
        f_in = load_json(data_file)
        f_in = f_in[:(len(f_in) // 128 * 16)]
        for line in tqdm(f_in):
            if 'subgraph' not in line:
                continue
            self.data.append(line)
            self.max_relevant_doc = max(self.max_relevant_doc, len(line['passages'])) if use_doc else None
            self.max_facts = max(self.max_facts, 2 * len(line['subgraph']['tuples']))
        print('max_relevant_doc: ', self.max_relevant_doc)
        print('max_facts: ', self.max_facts)
        self.num_data = len(self.data)
        self.batches = np.arange(self.num_data)

        print('building word index ...')
        self.word2id = word2id
        self.relation2id = relation2id
        self.entity2id = entity2id
        self.documents = documents
        self.id2entity = {i: entity for entity, i in entity2id.items()}

        print('converting global to local entity index ...')
        self.global2local_entity_maps = self._build_global2local_entity_maps()
        print('max_local_entity', self.max_local_entity)

        # print('indexing documents ...')
        self.document_entity_indices = document_entity_indices
        self.document_texts = document_texts

        print('preparing data ...')
        self.local_entities = np.full((self.num_data, self.max_local_entity), len(self.entity2id), dtype=int)
        self.kb_adj_mats = np.empty(self.num_data, dtype=object)
        self.kb_fact_rels = np.full((self.num_data, self.max_facts), self.num_kb_relation, dtype=int)
        self.q2e_adj_mats = np.zeros((self.num_data, self.max_local_entity, 1), dtype=float)
        self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        self.rel_document_ids = np.full((self.num_data, self.max_relevant_doc), -1,
                                        dtype=int) if use_doc else None  # the last document is empty
        self.entity_poses = np.empty(self.num_data, dtype=object)
        self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.fact_dropout = fact_dropout

        self._prepare_data()

    def _prepare_data(self):
        """
        global2local_entity_maps: a map from global entity id to local entity id
        adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
        """
        next_id = 0
        count_query_length = [0] * 50
        total_num_answerable_question = 0
        for sample in tqdm(self.data):
            # get a list of local entities
            g2l = self.global2local_entity_maps[next_id]
            for global_entity, local_entity in g2l.items():
                if local_entity != 0:  # skip question node
                    self.local_entities[next_id, local_entity] = global_entity

            entity2fact_e, entity2fact_f = [], []
            fact2entity_f, fact2entity_e = [], []

            entity_pos_local_entity_id = []
            entity_pos_word_id = []
            entity_pos_word_weights = []

            # relations in local KB
            if self.use_kb:
                for i, tpl in enumerate(sample['subgraph']['tuples']):
                    sbj, rel, obj = tpl
                    if sbj not in self.entity2id or obj not in self.entity2id or rel not in self.relation2id:
                        continue
                    if not self.use_inverse_relation:
                        entity2fact_e += [g2l[self.entity2id[sbj]]]
                        entity2fact_f += [i]
                        fact2entity_f += [i]
                        fact2entity_e += [g2l[self.entity2id[obj]]]
                        self.kb_fact_rels[next_id, i] = self.relation2id[rel]
                    else:
                        entity2fact_e += [g2l[self.entity2id[sbj]], g2l[self.entity2id[obj]]]
                        entity2fact_f += [2 * i, 2 * i + 1]
                        fact2entity_f += [2 * i, 2 * i + 1]
                        fact2entity_e += [g2l[self.entity2id[obj]], g2l[self.entity2id[sbj]]]
                        self.kb_fact_rels[next_id, 2 * i] = self.relation2id[rel]
                        self.kb_fact_rels[next_id, 2 * i + 1] = self.relation2id[rel] + len(self.relation2id)

            # build connection between question and entities in it
            for j, entity in enumerate(sample['entities']):
                if entity not in self.entity2id:
                    continue
                self.q2e_adj_mats[next_id, g2l[self.entity2id[entity]], 0] = 1.0

            # connect documents to entities occurred in it
            if self.use_doc:
                for j, passage in enumerate(sample['passages']):
                    document_id = passage['document_id']
                    if document_id not in self.document_entity_indices:
                        continue
                    (global_entity_ids, word_ids, word_weights) = self.document_entity_indices[document_id]
                    entity_pos_local_entity_id += [g2l[global_entity_id] for global_entity_id in global_entity_ids]
                    entity_pos_word_id += [word_id + j * self.max_document_word for word_id in word_ids]
                    entity_pos_word_weights += word_weights

            # tokenize question
            question_word_spt = clean_text(sample['question'])
            count_query_length[len(question_word_spt)] += 1
            for j, word in enumerate(question_word_spt):
                if j < self.max_query_word:
                    if word in self.word2id:
                        self.query_texts[next_id, j] = self.word2id[word]
                    else:
                        self.query_texts[next_id, j] = self.word2id['__unk__']

            # tokenize document
            if self.use_doc:
                for pid, passage in enumerate(sample['passages']):
                    self.rel_document_ids[next_id, pid] = passage['document_id']

            # construct distribution for answers
            for answer in sample['answers']:
                keyword = 'answer_id'
                if answer[keyword] in self.entity2id and self.entity2id[answer[keyword]] in g2l:
                    self.answer_dists[next_id, g2l[self.entity2id[answer[keyword]]]] = 1.0

            self.kb_adj_mats[next_id] = (np.array(entity2fact_f, dtype=int), np.array(entity2fact_e, dtype=int),
                                         np.array([1.0] * len(entity2fact_f))), (
                                        np.array(fact2entity_e, dtype=int), np.array(fact2entity_f, dtype=int),
                                        np.array([1.0] * len(fact2entity_e)))
            self.entity_poses[next_id] = (entity_pos_local_entity_id, entity_pos_word_id, entity_pos_word_weights)

            next_id += 1

    def _build_kb_adj_mat(self, sample_ids, fact_dropout):
        """Create sparse matrix representation for batched data"""
        if not isinstance(sample_ids, list):
            sample_ids = [sample_ids]
        mats0_batch = np.array([], dtype=int)
        mats0_0 = np.array([], dtype=int)
        mats0_1 = np.array([], dtype=int)
        vals0 = np.array([], dtype=float)

        mats1_batch = np.array([], dtype=int)
        mats1_0 = np.array([], dtype=int)
        mats1_1 = np.array([], dtype=int)
        vals1 = np.array([], dtype=float)

        for i, sample_id in enumerate(sample_ids):
            (mat0_0, mat0_1, val0), (mat1_0, mat1_1, val1) = self.kb_adj_mats[sample_id]
            assert len(val0) == len(val1)
            num_fact = len(val0)
            num_keep_fact = int(np.floor(num_fact * (1 - fact_dropout)))
            mask_index = np.random.permutation(num_fact)[: num_keep_fact]
            # mat0
            mats0_batch = np.append(mats0_batch, np.full(len(mask_index), i, dtype=int))
            mats0_0 = np.append(mats0_0, mat0_0[mask_index])
            mats0_1 = np.append(mats0_1, mat0_1[mask_index])
            vals0 = np.append(vals0, val0[mask_index])
            # mat1
            mats1_batch = np.append(mats1_batch, np.full(len(mask_index), i, dtype=int))
            mats1_0 = np.append(mats1_0, mat1_0[mask_index])
            mats1_1 = np.append(mats1_1, mat1_1[mask_index])
            vals1 = np.append(vals1, val1[mask_index])

        return (mats0_batch, mats0_0, mats0_1, vals0), (mats1_batch, mats1_0, mats1_1, vals1)

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # return self.local_entities[idx], \
        #        self.q2e_adj_mats[idx], \
        #        (self._build_kb_adj_mat(idx, fact_dropout=self.fact_dropout)), \
        #        self.kb_fact_rels[idx], \
        #        self.query_texts[idx], \
        #        self._build_document_text(idx), \
        #        (self._build_entity_pos(idx)), \
        #        self.answer_dists[idx]
        return self.local_entities[idx], \
               self.q2e_adj_mats[idx], \
               (self._build_kb_adj_mat(idx, fact_dropout=self.fact_dropout)), \
               self.kb_fact_rels[idx], \
               self.query_texts[idx], \
               self._build_document_text(idx), \
               (self._build_entity_pos(idx)), \
               self.answer_dists[idx]

    def _build_document_text(self, sample_ids):
        """Index tokenized documents for each sample"""
        if not self.use_doc:
            return None
        if not isinstance(sample_ids, list):
            sample_ids = [sample_ids]
        document_text = np.full((len(sample_ids), self.max_relevant_doc, self.max_document_word), len(self.word2id),
                                dtype=int)
        for i, sample_id in enumerate(sample_ids):
            for j, rel_doc_id in enumerate(self.rel_document_ids[sample_id]):
                if rel_doc_id not in self.document_texts:
                    continue
                document_text[i, j] = self.document_texts[rel_doc_id]
        return document_text

    def _build_entity_pos(self, sample_ids):
        """Index the position of each entity in documents"""
        if not isinstance(sample_ids, list):
            sample_ids = [sample_ids]
        entity_pos_batch = np.array([], dtype=int)
        entity_pos_entity_id = np.array([], dtype=int)
        entity_pos_word_id = np.array([], dtype=int)
        vals = np.array([], dtype=float)

        for i, sample_id in enumerate(sample_ids):
            (entity_id, word_id, val) = self.entity_poses[sample_id]
            num_nonzero = len(val)
            entity_pos_batch = np.append(entity_pos_batch, np.full(num_nonzero, i, dtype=int))
            entity_pos_entity_id = np.append(entity_pos_entity_id, entity_id)
            entity_pos_word_id = np.append(entity_pos_word_id, word_id)
            vals = np.append(vals, val)
        return (entity_pos_batch.astype(int), entity_pos_entity_id.astype(int), entity_pos_word_id.astype(int), vals)

    def _build_global2local_entity_maps(self):
        """Create a map from global entity id to local entity of each sample"""
        global2local_entity_maps = [None] * self.num_data
        total_local_entity = 0.0
        next_id = 0
        for sample in tqdm(self.data):
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
            self.max_local_entity = max(self.max_local_entity, len(g2l))
            next_id += 1
        print('avg local entity: ', total_local_entity / next_id)
        return global2local_entity_maps

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
