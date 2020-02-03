import itertools
import nltk
import numpy as np
from tqdm import tqdm
from util import load_json
from preprocessing.process_complex_webq import clean_text
from collections import defaultdict
from nltk.corpus import stopwords


KEEP_TAG = ['IN', 'NN', 'NNS', 'NNP', 'NNPS', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


class RelChainPredictorDataLoader():
    def __init__(self, data_file, facts, num_hop, word2id, relation2id, max_query_word, use_inverse_relation, div, teacher_force=False):
        self.use_inverse_relation = use_inverse_relation
        self.max_local_entity = 0
        self.max_facts = 0
        self.max_query_word = max_query_word
        self.facts = facts
        self.num_kb_relation = len(relation2id)
        self.teacher_force = teacher_force
        self.num_hop = num_hop
        self.max_local_path_rel = 300
        self.max_rel_num = 3

        self.data = []

        print('loading data from', data_file)
        f_in = load_json(data_file)
        f_in = f_in[:len(f_in)//div]
        for line in tqdm(f_in):
            if len(line['rel_chaim_map']) == 0:
                continue
            #self.max_local_path_rel = max(self.max_local_path_rel, len(line['rel_cands_multi_cands']))
            self.data.append(line)
        print('max_local_path_rel:', self.max_local_path_rel)
        self.data = np.array(self.data)
        self.num_data = len(self.data)

        self.batches = np.arange(self.num_data)

        print('building word index ...')
        self.word2id = word2id
        self.relation2id = relation2id

        print('preparing data ...')
        self.local_kb_rel_path_rels = np.full((self.num_data, self.max_local_path_rel), len(self.relation2id), dtype=int)
        self.relation_texts = np.full((self.num_data, self.max_local_path_rel, self.max_rel_num, self.max_query_word), len(self.word2id), dtype=int)
        self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        self.answer_dists = np.zeros((self.num_data, self.max_local_path_rel), dtype=float)

        self._prepare_data()

    def _prepare_data(self):
        """
        global2local_entity_maps: a map from global entity id to local entity id
        adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
        """
        next_id = 0
        count_query_length = [0] * 50
        cache_question = {}
        cache_relation = {}


        question_word_list = 'who, when, what, where, how, which, why, whom, whose, the'.split(', ')
        stop_words = set(stopwords.words("english"))
        for idx_q, sample in tqdm(enumerate(self.data)):
            # build answers


            # rel_ids = set()
            # if 'rel_cands_multi' in sample:
            #     # for hop in range(len(sample['rel_cands_multi'])):
            #     for rel in sample['rel_cands_multi'][min(len(sample['rel_cands_multi']), self.num_hop) - 1]:
            #         if 'common' in rel:
            #             continue
            #         rel_ids.add(self.relation2id[rel])
            # else:
            #     for rel in sample['rel_path']:
            #         rel_ids.add(self.relation2id[rel])
            #
            # rel_cands = sample['rel_cands_multi_cands']
            #
            # # filtering invalid cands
            # rel_cands = [e for e in rel_cands if (e not in (
            #     'Equals', 'GreaterThan', 'GreaterThanOrEqual', 'LessThan', 'LessThanOrEqual', 'NotEquals',
            #     'rdf-schema#domain', 'rdf-schema#range') and
            #                                       not e.startswith('freebase.'))]


            # for i in range(min(len(rel_cands), self.max_local_path_rel)):
            #     rel = rel_cands[i]
            #     rel_spt = rel.split('.')
            #     if len(rel_spt) > 3:
            #         rel_spt = rel_spt[-3:]
            #     for j in range(len(rel_spt)):
            #         if j - 3 < -len(rel_spt):
            #             rel = '__unk__'
            #         else:
            #             rel = rel_spt[j-3]
            #         if rel not in cache_relation:
            #             cache_relation[rel] = clean_text(rel, filter_dot=True)
            #         rel_word_spt = cache_relation[rel]
            #         for k, word in enumerate(rel_word_spt):
            #             if k < self.max_query_word:
            #                 if word in self.word2id:
            #                     self.relation_texts[idx_q, i, j, k] = self.word2id[word]
            #                 else:
            #                     self.relation_texts[idx_q, i, j, k] = self.word2id['__unk__']
            #     self.local_kb_rel_path_rels[idx_q][i] = self.relation2id[rel_cands[i]]
            #     if self.local_kb_rel_path_rels[idx_q][i] in rel_ids:
            #         self.answer_dists[idx_q][i] = 1.0


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
            for j, word in enumerate(question_word_spt):
                if j < self.max_query_word:
                    if word in self.word2id:
                        self.query_texts[next_id, j] = self.word2id[word]
                    else:
                        self.query_texts[next_id, j] = self.word2id['__unk__']

            next_id += 1

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

        return self.data[sample_ids], \
               self.query_texts[sample_ids], \
               self.relation_texts[sample_ids], \
               self.local_kb_rel_path_rels[sample_ids], \
               self.answer_dists[sample_ids]