import itertools
import nltk
import numpy as np
import copy
from tqdm import tqdm
from util import load_json
from preprocessing.process_complex_webq import clean_text
from collections import defaultdict
from nltk.corpus import stopwords


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

        self.data = []
        self.origin_data = []

        print('loading data from', data_file)
        if data:
            f_in = data
        else:
            f_in = load_json(data_file)
            f_in = f_in[:len(f_in)//div]
        for line in tqdm(f_in):
            if 'rel_chain_map' not in line or not line['rel_chain_map'] or not line['answers']:
                continue
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
        self.data = np.array(self.data, dtype=np.object)
        self.num_data = len(self.data)

        self.batches = np.arange(self.num_data)

        print('building word index ...')
        self.word2id = word2id
        self.relation2id = relation2id
        self.max_local_path_rel = len(self.relation2id)
        print('max_local_path_rel:', self.max_local_path_rel)
        self.reverse_relation2id = {v: k for k, v in relation2id.items()}

        print('preparing data ...')
        self.seed_entity_types = np.full((self.max_type_word, self.num_data), len(self.word2id), dtype=int)
        self.query_texts = np.full((self.max_query_word, self.num_data), len(self.word2id), dtype=int)
        self.targets = np.full((self.num_hop + 1, self.num_data), len(relation2id), dtype=int)

        self._prepare_data()

    def _prepare_data(self):
        """
        global2local_entity_maps: a map from global entity id to local entity id
        adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
        """
        next_id = 0
        count_query_length = [0] * 50
        cache_question = {}

        question_word_list = 'who, when, what, where, how, which, why, whom, whose, the'.split(', ')
        stop_words = set(stopwords.words("english"))
        avg_total_words_recall = 0.0
        avg_question_words_recall = 0.0
        for idx_q, sample in tqdm(enumerate(self.data)):
            # build answers
            target_answer_path = list()
            seed_entity = sample['seed_entity']
            ground_truth_key = 'rel_chain_ground_truth'
            if ground_truth_key in sample:
                gt = sample[ground_truth_key]
                for each_gt in gt:
                    target_answer_path = [self.relation2id[r] for r in each_gt]
                    break

            # build seed entity types
            total_words = 0
            known_words = 0
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
                    for idx, word in enumerate(seed_entity_type):
                        if idx >= self.max_type_word:
                            break
                        if word in self.word2id:
                            self.seed_entity_types[idx, next_id] = self.word2id[word]
                            known_words += 1
                        else:
                            self.seed_entity_types[idx, next_id] = self.word2id['__unk__']
                        total_words += 1
            avg_total_words_recall += ((known_words / total_words) if total_words > 0 else 1)

            self.targets[0, next_id] = self.relation2id['SOS']
            for j, target_rel_id in enumerate(target_answer_path):
                if j < self.num_hop:
                    self.targets[j + 1, next_id] = target_rel_id

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
            avg_qw_recall = 0
            for j, word in enumerate(question_word_spt):
                if j < self.max_query_word:
                    if word in self.word2id:
                        self.query_texts[j, next_id] = self.word2id[word]
                        avg_qw_recall += 1
                    else:
                        self.query_texts[j, next_id] = self.word2id['__unk__']
            avg_qw_recall /= len(question_word_spt)
            next_id += 1
            avg_question_words_recall += avg_qw_recall
        print('avg_total_words_recall', avg_total_words_recall / len(self.data))
        print('avg_question_words_recall', avg_question_words_recall / len(self.data))


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

        return self.query_texts[:, sample_ids], \
               self.seed_entity_types[:, sample_ids], \
               self.targets[:, sample_ids], \
               self.data[sample_ids]