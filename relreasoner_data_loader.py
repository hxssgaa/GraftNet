import itertools
import nltk
import numpy as np
import copy
from tqdm import tqdm
from util import load_json
from preprocessing.process_complex_webq import clean_text
from collections import defaultdict
from nltk.corpus import stopwords
from transformers import *


MAX_FACTS = 500
KEEP_TAG = ['IN', 'NN', 'NNS', 'NNP', 'NNPS', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


class RelReasonerDataLoader():
    def __init__(self, data_file, facts, features, num_hop, word2id, relation2id, max_query_word, use_inverse_relation, div, teacher_force=False, data=None):
        self.use_inverse_relation = use_inverse_relation
        self.max_local_entity = 0
        self.max_facts = 0
        self.max_query_word = max_query_word
        self.facts = facts
        self.features = features
        self.num_kb_relation = len(relation2id)
        self.teacher_force = teacher_force
        self.num_hop = num_hop
        self.max_local_path_rel = 0
        self.max_rel_num = 3
        self.max_seed_entities = 10
        self.max_type_word = 3
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        self.data = []
        self.origin_data = []

        print('loading data from', data_file)
        if data:
            f_in = data
        else:
            f_in = load_json(data_file)
            f_in = f_in[:len(f_in)//div]
        for line in tqdm(f_in):
            self.origin_data.append(line)
            rel_chain_map = line['rel_chain_map'][str(self.num_hop)]
            for k, v in rel_chain_map.items():
                data_line_copy = line.copy()
                data_line_copy['seed_entity'] = k
                gt = v['ground_truth'].copy()
                cands = v['cands'].copy()
                data_line_copy['rel_chain_ground_truth'] = gt
                data_line_copy['rel_chain_cands'] = cands
                self.data.append(data_line_copy)
                self.max_local_path_rel = max(self.max_local_path_rel, len(data_line_copy['rel_chain_cands']))
        print('max_local_path_rel:', self.max_local_path_rel)
        self.num_data = len(self.data)

        self.batches = np.arange(self.num_data)

        print('building word index ...')
        self.word2id = word2id
        self.relation2id = relation2id
        self.reverse_relation2id = {v: k for k, v in relation2id.items()}

        print('preparing data ...')
        self.seed_entity_types = np.full((self.num_data, self.max_type_word), self.tokenizer.unk_token_id, dtype=int)
        self.local_kb_rel_path_rels = np.full((self.num_data, self.max_local_path_rel, self.num_hop), len(self.relation2id), dtype=int)
        self.relation_texts = np.full((self.num_data, self.max_local_path_rel, self.max_rel_num, self.max_query_word), len(self.word2id), dtype=int)
        self.query_texts = np.full((self.num_data, self.max_query_word), self.tokenizer.unk_token_id, dtype=int)
        self.answer_dists = np.zeros((self.num_data, self.max_local_path_rel), dtype=float)

        self._prepare_data()

    def get_gt_constraint_rels(self, q, num_hop, no_compare_seed=False):
        if num_hop == 1:
            gt_constraint_rels = set((e[1],) for e in q['ground_truth_path'] if (num_hop - 1) * 2 < len(e)
                                     and num_hop * 2 - 1 < len(e) and (e[(num_hop - 1) * 2] == q['seed_entity'] or no_compare_seed))
        elif num_hop == 2:
            gt_constraint_rels = set((e[1], e[3]) for e in q['ground_truth_path'] if (num_hop - 1) * 2 < len(e)
                                     and num_hop * 2 - 1 < len(e) and (e[(num_hop - 1) * 2] == q['seed_entity'] or no_compare_seed))
        elif num_hop == 3:
            gt_constraint_rels = set((e[1], e[3], e[5]) for e in q['ground_truth_path'] if (num_hop - 1) * 2 < len(e)
                                     and num_hop * 2 - 1 < len(e) and (e[(num_hop - 1) * 2] == q['seed_entity'] or no_compare_seed))
        else:
            gt_constraint_rels = None
        return gt_constraint_rels

    def resample_multi_rels(self, q, num_hop, prev_rels):
        if 'ground_truth_map' not in q:
            return
        constraint_rels = set(self.facts[q['seed_entity']].keys())
        filter_rels = {'Equals', 'GreaterThan', 'GreaterThanOrEqual', 'LessThan', 'LessThanOrEqual', 'NotEquals'}
        constraint_rels -= filter_rels
        gt_constraint_rels = self.get_gt_constraint_rels(q, num_hop)
        if not constraint_rels:
            constraint_rels = {'EOD'}
        if prev_rels:
            constraint_rels = set([tuple(prev_rels + [e2]) for e2 in constraint_rels])
        # TODO: FIXTHIS for multi-hop
        if num_hop - 1 < len(q['rel_cands_multi']):
            q['rel_cands_multi'][num_hop - 1] = list(gt_constraint_rels)
            if not gt_constraint_rels:
                del q['rel_cands_multi'][num_hop - 1]
        q['rel_cands_multi_cands%d' % num_hop] = list(constraint_rels)

    def _prepare_data(self):
        """
        global2local_entity_maps: a map from global entity id to local entity id
        adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
        """
        next_id = 0
        count_query_length = [0] * 50
        cache_question = {}

        # for rel in self.relation2id:
        #     idx_rel = self.relation2id[rel]
        #     rel_word_spt = clean_text(rel)
        #     for j, word in enumerate(rel_word_spt):
        #         if j < self.max_query_word:
        #             if word in self.word2id:
        #                 self.relation_texts[idx_rel, j] = self.word2id[word]
        #             else:
        #                 self.relation_texts[idx_rel, j] = self.word2id['__unk__']

        question_word_list = 'who, when, what, where, how, which, why, whom, whose, the'.split(', ')
        stop_words = set(stopwords.words("english"))
        avg_total_words_recall = 0.0
        for idx_q, sample in tqdm(enumerate(self.data)):
            # build answers
            rel_ids = set()
            seed_entity = sample['seed_entity']
            ground_truth_key = 'rel_chain_ground_truth'
            if ground_truth_key in sample:
                gt = sample[ground_truth_key]
                for each_gt in gt:
                    rel_id = tuple([self.relation2id[r] for r in each_gt])
                    rel_ids.add(rel_id)

            # if 'rel_cands_multi' in sample:
            #     if self.num_hop - 1 < len(sample['rel_cands_multi']):
            #         for rel in sample['rel_cands_multi'][self.num_hop - 1]:
            #             rel_ids.add(tuple([self.relation2id[r] for r in rel]))
            #     else:
            #         prev_rels = set()
            #         if len(sample['rel_cands_multi']) > self.num_hop - 2 and self.num_hop > 1:
            #             for rel in sample['rel_cands_multi'][self.num_hop - 2]:
            #                 prev_rels.add(tuple(rel))
            #         elif len(sample['rel_cands_multi']) > self.num_hop - 3 and self.num_hop > 2:
            #             for rel in sample['rel_cands_multi'][self.num_hop - 3]:
            #                 prev_rels.add(tuple(rel))
            #         for prev_rel in prev_rels:
            #             rel_id_tuple = tuple([self.relation2id[r] for r in prev_rel] + [self.relation2id['EOD']])
            #             if len(rel_id_tuple) < self.num_hop:
            #                 rel_id_tuple = tuple(list(rel_id_tuple) + [self.relation2id['EOD']] * (self.num_hop - len(rel_id_tuple)))
            #             rel_ids.add(rel_id_tuple)

            # build seed entity types
            seed_entity_type = None
            if seed_entity in self.features:
                if 'prom_types' in self.features[seed_entity] and self.features[seed_entity]['prom_types']:
                    seed_entity_type = self.features[seed_entity]['prom_types'][0] \
                        if isinstance(self.features[seed_entity]['prom_types'], list) else self.features[seed_entity]['prom_types']
                elif 'types' in self.features[seed_entity] and self.features[seed_entity]['types']:
                    seed_entity_type = self.features[seed_entity]['types'][0] \
                        if isinstance(self.features[seed_entity]['types'], list) else self.features[seed_entity]['types']
                if seed_entity_type:
                    seed_entity_type = seed_entity_type.split('.')[-1].split('_')
                    seed_entity_type = [(self.word2id[e] if e in self.word2id else self.word2id['__unk__']) for e in seed_entity_type]
                    for idx, word in enumerate(seed_entity_type):
                        if idx >= self.max_type_word:
                            break
                        # if word in self.word2id:
                        #     self.seed_entity_types[next_id, idx] = self.word2id[word]
                        # else:
                        #     self.seed_entity_types[next_id, idx] = self.word2id['__unk__']
                        self.seed_entity_types[next_id, idx] = word

            # get a list of relation candidate paths
            # for i, rel in enumerate(rel_ids):
            #     self.local_kb_rel_path_rels[idx_q][i] = rel
            #     self.answer_dists[idx_q][i] = 1.0
            #
            # rel_cands = sample['rel_cands_multi_cands']
            # for i in range(len(rel_ids), min(self.max_local_path_rel, len(rel_cands))):
            #     self.local_kb_rel_path_rels[idx_q][i] = self.relation2id[rel_cands[i]]

            rel_cands = sample['rel_chain_cands']
            # if self.num_hop > 1:
            #     prev_hop_rels = set()
            #     for e in rel_cands:
            #         prev_hop_rels.add(tuple(e[:self.num_hop - 1]))
            #     for e in prev_hop_rels:
            #         rel_cands.insert(0, list(e) + ['EOD'])
            #     if not rel_cands and 'relation_pred2' in sample:
            #         rel_cands = [list(sample['relation_pred2']) + ['EOD']]

            # # filtering invalid cands
            # rel_cands = [e for e in rel_cands if (e[-1] not in (
            #     'Equals', 'GreaterThan', 'GreaterThanOrEqual', 'LessThan', 'LessThanOrEqual', 'NotEquals',
            #     'rdf-schema#domain', 'rdf-schema#range') and
            #                                       not e[-1].startswith('freebase.'))]

            # if self.teacher_force:
            #     rel_cands_set = list(gt_rels | set(rel_cands[:self.max_local_path_rel - len(gt_rels)]))
            #     rel_cands = rel_cands_set
            # else:
            #     self.max_local_path_rel = 300

            for i in range(min(len(rel_cands), self.max_local_path_rel)):
                for j in range(self.num_hop):
                    self.local_kb_rel_path_rels[idx_q][i][j] = self.relation2id[rel_cands[i][j]]
                    key = tuple(self.local_kb_rel_path_rels[idx_q][i])
                    if key in rel_ids:
                        self.answer_dists[idx_q][i] = 1.0

            if np.sum(self.answer_dists[idx_q]) == 0:
                pass

            # if np.sum(self.answer_dists[idx_q]) == 0:
            #     print('wow')

            # for i in range(len(rel_cands)):
            #     rel_cand = list(map(self.relation2id.get, rel_cands[i]))
            #     if tuple(rel_cand) in rel_ids:
            #         self.answer_dists[idx_q][i] = 1.0
            #     for hop in range(self.num_hop):
            #         self.local_kb_rel_path_rels[idx_q][i][hop] = rel_cand[hop]

            stop_words.update(question_word_list)

            # tokenize question
            if sample['question'] in cache_question:
                question_word_spt = cache_question[sample['question']]
            else:
                tokens = nltk.word_tokenize(sample['question'])
                tagged_sent = nltk.pos_tag(tokens)
                grammar = "NP: {<DT><NN*>+<IN><NNP>+}"
                cp = nltk.RegexpParser(grammar)
                parsed = cp.parse(tagged_sent)
                question_word_spt = []
                for e in parsed:
                    if isinstance(e, nltk.Tree):
                        for sub in e:
                            if sub[1] in KEEP_TAG:
                                question_word_spt.append(sub[0])
                    elif e[1] in KEEP_TAG:
                        question_word_spt.append(e[0])
                # question_word_spt = clean_text(sample['question'])
                # question_word_spt = list(filter(lambda x: x not in question_word_list, question_word_spt))
                # question_word_spt = [e[0] for e in parsed if e[1] in KEEP_TAG]
                cache_question[sample['question']] = question_word_spt
            count_query_length[len(question_word_spt)] += 1
            question_word_spt = [(self.word2id[e] if e in self.word2id else self.word2id['__unk__']) for e in
                                 question_word_spt]
            recall = 0
            for j, word in enumerate(question_word_spt):
                if j < self.max_query_word:
                    # if word in self.word2id:
                    #     self.query_texts[next_id, j] = self.word2id[word]
                    # else:
                    #     self.query_texts[next_id, j] = self.word2id['__unk__']
                    self.query_texts[next_id, j] = word
                    if word != self.tokenizer.unk_token_id:
                        recall += 1
            recall /= min(self.max_query_word, len(question_word_spt))
            avg_total_words_recall += recall

            next_id += 1
        print(avg_total_words_recall / len(self.data))
        # print('asdasdasdas')

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
               self.relation_texts[sample_ids], \
               self.local_kb_rel_path_rels[sample_ids], \
               self.seed_entity_types[sample_ids], \
               self.answer_dists[sample_ids]