import numpy as np
import json
from tqdm import tqdm

from preprocessing.process_complex_webq import clean_text
from util import load_json


class DataLoader():
    def __init__(self, data_file, documents, document_entity_indices, document_texts, word2id,
                 relation2id, max_query_word, max_document_word, use_kb, use_doc, use_inverse_relation):
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

        self.data = []

        print('loading data from', data_file)
        self.data = []
        f_in = load_json(data_file)
        f_in = f_in[:len(f_in) // 10]
        for line in tqdm(f_in):
            self.data.append(line)
        self.data = np.array(self.data)
        self.num_data = len(self.data)
        self.batches = np.arange(self.num_data)

        print('building word index ...')
        self.word2id = word2id
        self.relation2id = relation2id
        self.documents = documents
        # print('indexing documents ...')
        self.document_entity_indices = document_entity_indices
        self.document_texts = document_texts

        print('preparing data ...')
        self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        self.rel_document_ids = np.full((self.num_data, self.max_relevant_doc), -1,
                                        dtype=int) if use_doc else None  # the last document is empty

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

            next_id += 1

    def _build_kb_adj_mat(self, sample_ids, fact_dropout):
        """Create sparse matrix representation for batched data"""
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

    def reset_batches(self, is_sequential=True):
        if is_sequential:
            self.batches = np.arange(self.num_data)
        else:
            self.batches = np.random.permutation(self.num_data)

    def get_batch(self, iteration, batch_size, fact_dropout):
        """
        *** return values ***
        :local_entity: global_id of each entity (batch_size, max_local_entity)
        :adj_mat: adjacency matrices (batch_size, num_relation, max_local_entity, max_local_entity)
        :query_text: a list of words in the query (batch_size, max_query_word)
        :rel_document_ids: (batch_size, max_relevant_doc)
        :answer_dist: an distribution over local_entity (batch_size, max_local_entity)
        """
        sample_ids = self.batches[batch_size * iteration: batch_size * (iteration + 1)]

        return self.query_texts[sample_ids], \
               self.data[sample_ids]

    def _build_document_text(self, sample_ids):
        """Index tokenized documents for each sample"""
        if not self.use_doc:
            return None
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

