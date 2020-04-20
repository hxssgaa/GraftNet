import itertools

import numpy as np
from tqdm import tqdm
from util import load_json
from preprocessing.process_complex_webq import clean_text
from collections import defaultdict


MAX_FACTS = 500


class RelReasonerDataLoader():
    def __init__(self, data_file, facts, num_hop, word2id, relation2id, max_query_word, use_inverse_relation, div, teacher_force=False):
        self.use_inverse_relation = use_inverse_relation
        self.max_local_entity = 0
        self.max_facts = 0
        self.max_query_word = max_query_word
        self.facts = facts
        self.num_kb_relation = len(relation2id)
        self.teacher_force = teacher_force
        self.num_hop = num_hop

        self.data = []

        print('loading data from', data_file)
        f_in = load_json(data_file)
        f_in = f_in[:len(f_in)//div]
        for line in tqdm(f_in):
            self.data.append(line)
        print('div', div, len(self.data))
        self.num_data = len(self.data)
        self.max_rel_paths = self.num_kb_relation ** num_hop
        print('max_rel_paths', self.max_rel_paths)

        self.batches = np.arange(self.num_data)

        print('building word index ...')
        self.word2id = word2id
        self.relation2id = relation2id

        print('preparing data ...')
        self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)

        print('preparing data ...')
        #self.local_kb_rel_paths = np.full((self.num_data, self.max_rel_paths), self.num_kb_relation, dtype=int)
        self.local_kb_rel_path_rels = np.zeros((self.num_data, self.max_rel_paths, num_hop))
        self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        self.answer_dists = np.zeros((self.num_data, self.max_rel_paths), dtype=float)

        self._prepare_data()

    def _prepare_data(self):
        """
        global2local_entity_maps: a map from global entity id to local entity id
        adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
        """
        next_id = 0
        count_query_length = [0] * 50
        cache_question = {}
        for idx_q, sample in tqdm(enumerate(self.data)):
            # build answers
            rel_ids = set()
            for rel_path in sample['rel_path']:
                tmp = []
                for p in rel_path:
                    rel_id = self.relation2id[p]
                    tmp.append(rel_id)
                rel_ids.add(tuple(tmp))

            # get a list of relation candidate paths
            rel_cands = [p for p in itertools.product(list(range(self.num_kb_relation)), repeat=3)]
            for i in range(self.max_rel_paths):
                if tuple(rel_cands[i]) in rel_ids:
                    self.answer_dists[idx_q][i] = 1.0
                for hop in range(self.num_hop):
                    self.local_kb_rel_path_rels[idx_q][i][hop] = rel_cands[i][hop]

            # tokenize question
            if sample['question'] in cache_question:
                question_word_spt = cache_question[sample['question']]
            else:
                question_word_spt = clean_text(sample['question'])
                cache_question[sample['question']] = question_word_spt
            count_query_length[len(question_word_spt)] += 1
            for j, word in enumerate(question_word_spt):
                if j < self.max_query_word:
                    if word in self.word2id:
                        self.query_texts[next_id, j] = self.word2id[word]
                    else:
                        self.query_texts[next_id, j] = self.word2id['__unk__']

            next_id += 1

    def _build_local_tuples(self, seeds, answers):
        # relations in local KB
        tuples = set()
        entities = set()
        if self.teacher_force:
            for entity in seeds:
                entities.add(entity)
                for answer in answers:
                    if isinstance(answer, dict):
                        answer = answer['text']
                    if answer in self.facts[entity]:
                        entities.add(answer)
                        rel, direction = self.facts[entity][answer]
                        if direction == 0:
                            tuples.add((entity, rel, answer))
                        else:
                            tuples.add((answer, rel, entity))

        flag = False
        for entity in seeds:
            if flag:
                break
            for neighbor in list(self.facts[entity].keys()):
                rel, direction = self.facts[entity][neighbor]
                if len(tuples) >= MAX_FACTS:
                    flag = True
                    break
                entities.add(neighbor)
                if direction == 0:
                    tuples.add((entity, rel, neighbor))
                else:
                    tuples.add((neighbor, rel, entity))

        return {'entities': list(entities), 'tuples': list(tuples)}

    def build_relreasoner_data(self, data_file):
        res = []
        f_in = load_json(data_file)
        f_in = f_in[:len(f_in)]
        for line in tqdm(f_in):
            seed2next = defaultdict(list)
            for path in line['path']:
                if len(path) >= 3:
                    seed2next[path[1]].append(path[2])

            for seed in seed2next:
                answers = seed2next[seed]
                seeds = [seed]
                seeds = list(map(lambda x: x.replace('%', ''), seeds))
                p_data = line.copy()
                p_data['entities'] = seeds
                p_data['answers'] = answers
                p_data['hop'] = 0
                p_data['subgraph'] = self._build_local_tuples(seeds, answers)
                self.max_facts = max(self.max_facts, 2 * len(p_data['subgraph']['tuples']))
                res.append(p_data)
        return np.array(res)

    def reset_batches(self, is_sequential=True):
        if is_sequential:
            self.batches = np.arange(self.num_data)
        else:
            self.batches = np.random.permutation(self.num_data)

    @staticmethod
    def _add_entity_to_map(entity2id, entities, g2l):
        for entity in entities:
            if isinstance(entity, dict):
                entity_text = entity['text']
            else:
                entity_text = entity
            entity_global_id = entity2id[entity_text]
            if entity_global_id not in g2l:
                g2l[entity2id[entity_text]] = len(g2l)

    def _build_global2local_entity_maps(self):
        """Create a map from global entity id to local entity of each sample"""
        global2local_entity_maps = [None] * self.num_data
        total_local_entity = 0.0
        next_id = 0
        for sample in tqdm(self.data):
            g2l = dict()
            self._add_entity_to_map(self.entity2id, sample['entities'], g2l)
            # construct a map from global entity id to local entity id
            self._add_entity_to_map(self.entity2id, sample['subgraph']['entities'], g2l)

            global2local_entity_maps[next_id] = g2l
            total_local_entity += len(g2l)
            self.max_local_entity = max(self.max_local_entity, len(g2l))
            next_id += 1
        print('avg local entity: ', total_local_entity / next_id)
        return global2local_entity_maps


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
            mask_index = np.random.permutation(num_fact)[ : num_keep_fact]
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
               self.local_kb_rel_path_rels[sample_ids], \
               self.answer_dists[sample_ids]