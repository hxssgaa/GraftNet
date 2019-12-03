import json
import os
from multiprocessing.pool import Pool
from string import punctuation
from functools import partial

import numpy as np
import wordninja
import pickle as pkl
from scipy.sparse import csr_matrix
from tqdm import tqdm, trange
from preprocessing.use_helper import UseVector
from localgraphclustering import *
from sklearn.preprocessing import normalize

QUESTION_FILE = 'datasets/complexwebq/questions/all_questions_v3_hop3.json'
FACTS_FILE = 'datasets/complexwebq/all_facts.json'
RELATIONS_FILE = 'datasets/complexwebq/relations.txt'
OUT_QUESTION_EMBEDDING = 'datasets/complexwebq/all_questions_embeddings_v2.pkl'
OUT_RELATIONS_EMBEDDING = 'datasets/complexwebq/relation_emb_v2.pkl'

MAX_FACTS = 5000000
MAX_ITER = 2
RESTART = 0.8
NOTFOUNDSCORE = 0.
EXPONENT = 2.
MAX_ENT = 200
PARALLEL_PROCESSOR = 8


def clean_text(text, filter_dot=False):
    text = text.replace('.', ' . ').lower()
    for punc in punctuation:
        if punc != '.':
            text = text.replace(punc, " ")
    text = text.split()
    output = []
    for i in text:
        if len(i) < 10:
            output.append(i)
        else:
            output.extend(wordninja.split(i))
    if filter_dot:
        return [e for e in text if e != '.']
    return text


def save_json(list_, name):
    with open(name, 'w') as f:
        json.dump(list_, f)


def load_json(file):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise e
    return data


def process_entities():
    data = load_json('datasets/complexwebq/train.json')
    entities = set()
    for q in data:
        entities.update(q['subgraph']['entities'])
    # ground_truth_entities = set(load_json('datasets/complexwebq/ground_truth_entities.json'))
    # entities.update(ground_truth_entities)
    entities = list(sorted(entities))
    with open('datasets/complexwebq/entities.txt', 'w') as f:
        for e in entities:
            if e.startswith('m.') or e.startswith('g.'):
                f.writelines(e + '\n')


# def process_vocab():
#     vocab2id = load_json('datasets/complexwebq/vocab2id.json')
#     with open('datasets/complexwebq/vocab.txt', 'w') as f:
#         for e in vocab2id:
#             f.writelines(e + '\n')


def process_relations():
    relations = load_json('datasets/complexwebq/relations_new.json')
    with open('datasets/complexwebq/relations.txt', 'w') as f:
        for e in relations:
            f.writelines(e + '\n')


def fix_questions():
    base_path = 'datasets/complexwebq/m1'
    train_questions = load_json('datasets/complexwebq/questions/ComplexWebQuestions_train.json')
    dev_questions = load_json('datasets/complexwebq/questions/ComplexWebQuestions_dev.json')
    test_questions = load_json('datasets/complexwebq/questions/ComplexWebQuestions_test.json')
    train_qids = {'%s.json' % q['ID'] for q in train_questions}
    dev_qids = {'%s.json' % q['ID'] for q in dev_questions}
    test_qids = {'%s.json' % q['ID'] for q in test_questions}
    train_data = []
    dev_data = []
    test_data = []
    for file_name in os.listdir(base_path):
        file_path = os.path.join(base_path, file_name)
        q = load_json(file_path)
        if file_name in dev_qids:
            dev_data.append(q)
        elif file_name in test_qids:
            test_data.append(q)
        else:
            train_data.append(q)
    for q in train_data + dev_data + test_data:
        if 'answers' in q:
            continue
        answers = set()
        for path in q['path']:
            if len(path) % 2 != 0:
                answers.update(path[-1])
        q['answers'] = list(map(lambda x: {'answer_id': x}, sorted(answers)))
    print(len(train_data))
    print(len(dev_data))
    print(len(test_data))
    save_json(train_data, 'datasets/complexwebq/train.json')
    save_json(dev_data, 'datasets/complexwebq/dev.json')
    save_json(test_data, 'datasets/complexwebq/test.json')


def analyse_testing_recall():
    base_path = 'datasets/complexwebq/m1'
    test_questions = load_json('datasets/complexwebq/questions/ComplexWebQuestions_test.json')
    test_qids = {'%s.json' % q['ID'] for q in test_questions}
    avg_recall = 0
    total = 0
    for file_name in os.listdir(base_path):
        if file_name not in test_qids:
            continue
        file_path = os.path.join(base_path, file_name)
        q = load_json(file_path)
        avg_recall += q['subgraph']['recall']
        total += 1
    print(avg_recall / total)


def process_vocab(embeddings_file):
    word2vec = {}
    with open(embeddings_file) as f:
        for line in tqdm(f):
            word, vec = line.strip().split(None, 1)
            word2vec[word] = np.array([float(vv) for vv in vec.split()])
    train_data = load_json('datasets/complexwebq/train.json')
    dev_data = load_json('datasets/complexwebq/dev.json')
    test_data = load_json('datasets/complexwebq/test.json')
    corpus = set()
    dev_corpus = set()
    test_corpus = set()
    for q in train_data:
        processed_question = clean_text(q['question'])
        corpus.update(processed_question)
    for q in dev_data:
        processed_question = clean_text(q['question'])
        dev_corpus.update(processed_question)
    for q in test_data:
        processed_question = clean_text(q['question'])
        test_corpus.update(processed_question)
    corpus.add('__unk__')
    corpus_list = list(sorted(corpus))
    word_emb_npy = np.array(
        [word2vec[word] if word in word2vec else np.random.uniform(-1, 1, 100) for word in corpus_list])
    with open('datasets/complexwebq/vocab.txt', 'w') as f:
        for e in corpus_list:
            f.writelines(e + '\n')
    np.save('datasets/complexwebq/word_emb_100d.npy', word_emb_npy)
    print('train emb coverage: ', len(corpus & set(word2vec.keys())) / len(corpus))
    print('dev coverage: ', len(corpus & dev_corpus) / len(dev_corpus))
    print('test coverage: ', len(corpus & test_corpus) / len(test_corpus))


def question_embedding(questions, use_f=True):
    if os.path.exists(OUT_QUESTION_EMBEDDING) and use_f:
        return pkl.load(open(OUT_QUESTION_EMBEDDING, 'rb'))

    question_emb = {}
    use = UseVector()

    # Get the work2question map and the question word length map.
    for ii, question in tqdm(enumerate(questions), total=len(questions)):
        raw_text = question['question']
        q_id = question["ID"]
        q_emb = use.get_vector(raw_text)[0]
        question_emb[q_id] = q_emb

    pkl.dump(question_emb, open(OUT_QUESTION_EMBEDDING, "wb"))

    return question_emb


def relation_embeddings(use_f=True):
    if os.path.exists(OUT_RELATIONS_EMBEDDING) and use_f:
        return pkl.load(open(OUT_RELATIONS_EMBEDDING, 'rb'))

    relation_emb = {}
    use = UseVector()

    with open(RELATIONS_FILE) as f:
        relations = f.readlines()
    for ii, line in tqdm(enumerate(relations), total=len(relations)):
        relation = line.strip()
        if relation in ["GreaterThan", "LessThan", "NotEquals"]:
            print(relation)
            relation = 'common.' + relation + '.object'
        elif relation.count('.') < 2:
            print(relation)
            relation = 'common.notable_for.object'
        domain, typ, prop = relation.split(".")[-3:]
        relation_emb[relation] = (use.get_vector(domain)[0] + 2 * use.get_vector(typ)[0] + 3 * use.get_vector(prop)[
            0]) / 6

    pkl.dump(relation_emb, open(OUT_RELATIONS_EMBEDDING, "wb"))
    return relation_emb


def _personalized_pagerank(seed, W):
    """Return the PPR vector for the given seed and adjacency matrix.

    Args:
        seed: A sparse matrix of size E x 1.
        W: A sparse matrix of size E x E whose rows sum to one.

    Returns:
        ppr: A vector of size E.
    """
    restart_prob = RESTART
    r = restart_prob * seed
    s_ovr = np.copy(r)
    for i in range(MAX_ITER):
        r_new = (1. - restart_prob) * (W.transpose().dot(r))
        s_ovr = s_ovr + r_new
        delta = abs(r_new.sum())
        if delta < 1e-5: break
        r = r_new
    return np.squeeze(s_ovr)


def _filter_relation(relation):
    if relation == "<fb:common.topic.notable_types>": return False
    domain = relation[4:-1].split(".")[0]
    if domain == "type" or domain == "common": return True
    return False


def _convert_to_readable(tuples, inv_map):
    readable_tuples = []
    for tup in tuples:
        readable_tuples.append([
            inv_map[tup[0]],
            tup[1],
            inv_map[tup[2]],
        ])
    return readable_tuples

# facts = load_json('datasets/complexwebq/all_facts.json')

def m1(q_embs, r_embs, rel_indexer, qs):
    avg_recall = 0
    t = trange(len(qs), desc='recall', leave=True)
    total = 0
    for idx, q in tqdm(enumerate(qs)):
        try:
            q_emb = q_embs[q['ID']]
            if 'hop_3' not in q:
                keys = q['entities']
            else:
                keys = q['hop_3']

            new_keys = set()
            sources = []
            targets = []
            local_num_entities = 0
            local_entities_map = {}
            local_relation_map = {}
            for k1 in keys:
                if k1 not in local_entities_map:
                    local_entities_map[k1] = local_num_entities
                    local_num_entities += 1
                for k2 in facts[k1]:
                    if _filter_relation(facts[k1][k2][1]): continue
                    if facts[k1][k2][1] not in rel_indexer:
                        continue
                    if k2 not in local_entities_map:
                        local_entities_map[k2] = local_num_entities
                        local_num_entities += 1
                    new_keys.add(k2)
                    if facts[k1][k2][0] == 0:
                        sources.append(local_entities_map[k1])
                        targets.append(local_entities_map[k2])
                    else:
                        sources.append(local_entities_map[k2])
                        targets.append(local_entities_map[k1])
                    rel = facts[k1][k2][1]
                    # rel_idx = rel_indexer[rel]
                    # if rel in r_embs:
                    #     score = np.dot(q_emb, r_embs[rel]) / (
                    #                 np.linalg.norm(q_emb) * np.linalg.norm(r_emb[rel]))
                    # else:
                    #     score = 0.1
                    # weights.append(0.5)
                    if rel not in local_relation_map:
                        local_relation_map[rel] = [[], []]
                    local_relation_map[rel][0].append(sources[-1])
                    local_relation_map[rel][1].append(targets[-1])
                    local_relation_map[rel][0].append(targets[-1])
                    local_relation_map[rel][1].append(sources[-1])
            # keys = new_keys
            # new_keys = set()
            # for k1 in keys:
            #     if k1 not in local_entities_map:
            #         local_entities_map[k1] = local_num_entities
            #         local_num_entities += 1
            #     for k2 in second_fact[k1]:
            #         if second_fact[k1][k2][1] not in rel_indexer:
            #             continue
            #         if k2 not in local_entities_map:
            #             local_entities_map[k2] = local_num_entities
            #             local_num_entities += 1
            #         new_keys.add(k2)
            #         if second_fact[k1][k2][0] == 0:
            #             sources.append(local_entities_map[k1])
            #             targets.append(local_entities_map[k2])
            #         else:
            #             sources.append(local_entities_map[k2])
            #             targets.append(local_entities_map[k1])
            #         rel = second_fact[k1][k2][1]
            #         rel_idx = rel_indexer[rel]
            #         weights.append(scores[sidx + idx][rel_idx])
            #         if rel_idx not in local_relation_map:
            #             local_relation_map[rel_idx] = [[], []]
            #         local_relation_map[rel_idx][0].append(sources[-1])
            #         local_relation_map[rel_idx][1].append(targets[-1])
            # keys = new_keys
            # new_keys = set()
            #         for k1 in keys:
            #             if k1 not in local_entities_map:
            #                 local_entities_map[k1] = local_num_entities
            #                 local_num_entities += 1
            #             for k2 in third_fact[k1]:
            #                 if third_fact[k1][k2][1] not in rel_indexer:
            #                     continue
            #                 if k2 not in local_entities_map:
            #                     local_entities_map[k2] = local_num_entities
            #                     local_num_entities += 1
            #                 new_keys.add(k2)
            #                 if third_fact[k1][k2][0] == 0:
            #                     sources.append(local_entities_map[k1])
            #                     targets.append(local_entities_map[k2])
            #                 else:
            #                     sources.append(local_entities_map[k2])
            #                     targets.append(local_entities_map[k1])
            #                 rel = third_fact[k1][k2][1]
            #                 rel_idx = rel_indexer[rel]
            #                 weights.append(scores[sidx + idx][rel_idx])
            #                 if rel_idx not in local_relation_map:
            #                     local_relation_map[rel_idx] = [[], []]
            #                 local_relation_map[rel_idx][0].append(sources[-1])
            #                 local_relation_map[rel_idx][1].append(targets[-1])
            reverse_local_entities_map = {v: k for k, v in local_entities_map.items()}
            # g = GraphLocal()
            # if not sources:
            #     continue
            # g.list_to_gl(sources, targets, weights)
            # entities = list(map(local_entities_map.get, q['entities']))
            # res = approximate_PageRank_weighted(g, entities)
            hop = 2
            targets = set()
            for path in q['path']:
                if len(path) % 2 != 0:
                    # for x in path[hop * 2 if (hop * 2 < len(path)) else -1]:
                    for x in path[-1]:
                        if x in local_entities_map:
                            targets.add(local_entities_map[x])
            entities = [local_entities_map[key] for key in keys]
            # ppr = np.argsort(res[1])[::-1][:100]
            # extracted_ents = res[0][ppr]
            # extracted_scores = res[1][ppr]
            # extracted_tuples = []
            for rel in local_relation_map:
                row_ones, col_ones = local_relation_map[rel]
                m = csr_matrix((np.ones((len(row_ones),)), (np.array(row_ones), np.array(col_ones))),
                               shape=(local_num_entities, local_num_entities))
                local_relation_map[rel] = normalize(m, norm="l1", axis=1)
                if rel not in r_embs:
                    score = NOTFOUNDSCORE
                else:
                    score = np.dot(q_emb, r_embs[rel]) / (
                            np.linalg.norm(q_emb) *
                            np.linalg.norm(r_embs[rel]))
                local_relation_map[rel] = local_relation_map[rel] * np.power(score, EXPONENT)
            adj_mat = sum(local_relation_map.values()) / len(local_relation_map)

            seed = np.zeros((adj_mat.shape[0], 1))
            seed[entities] = np.expand_dims(np.arange(len(entities), 0, -1),
                                            axis=1)
            seed = seed / seed.sum()
            ppr = _personalized_pagerank(seed, adj_mat)
            sorted_idx = np.argsort(ppr)[::-1]
            extracted_ents = sorted_idx[:MAX_ENT]
            extracted_scores = ppr[sorted_idx[:MAX_ENT]]
            # check if any ppr values are nearly zero
            zero_idx = np.where(ppr[extracted_ents] < 1e-6)[0]
            if zero_idx.shape[0] > 0:
                extracted_ents = extracted_ents[:zero_idx[0]]
            extracted_tuples = []
            ents_in_tups = set()
            for relation in local_relation_map:
                submat = local_relation_map[relation][extracted_ents, :]
                submat = submat[:, extracted_ents]
                row_idx, col_idx = submat.nonzero()
                for ii in range(row_idx.shape[0]):
                    extracted_tuples.append(
                        (extracted_ents[row_idx[ii]], relation,
                         extracted_ents[col_idx[ii]]))
                    ents_in_tups.add((extracted_ents[row_idx[ii]],
                                      extracted_scores[row_idx[ii]]))
                    ents_in_tups.add((extracted_ents[col_idx[ii]],
                                      extracted_scores[col_idx[ii]]))
                # submat = m[extracted_ents, :]
                # submat = submat[:, extracted_ents]
                # row_idx, col_idx = submat.nonzero()
                # for ii in range(row_idx.shape[0]):
                #     extracted_tuples.append(
                #         (reverse_local_entities_map[extracted_ents[row_idx[ii]]], reverse_local_relations_map[rel],
                #          reverse_local_entities_map[extracted_ents[col_idx[ii]]]))
            # extracted_ent_sets = set(extracted_ents)

            subgraph_entities = set(q['subgraph']['entities']) if 'subgraph' in q and 'entities' in q[
                'subgraph'] else set()
            subgraph_tuples = set(map(tuple, q['subgraph']['tuples'])) if 'subgraph' in q and 'tuples' in q[
                'subgraph'] else set()

            extracted_ents = list(map(reverse_local_entities_map.get, extracted_ents))
            extracted_scores = list(map(float, extracted_scores))

            subgraph_entities.update(extracted_ents)
            subgraph_tuples.update(list(map(tuple, _convert_to_readable(extracted_tuples, reverse_local_entities_map))))
            targets = set(map(reverse_local_entities_map.get, targets))


            if not targets:
                recall = 0
            else:
                recall = len((set(extracted_ents)) & targets) / len(targets)
            avg_recall += recall
            total += 1

            q['subgraph'] = {
                'entities': list(subgraph_entities),
                'scores': extracted_scores,
                'tuples': list(subgraph_tuples),
                'recall': recall
            }

            # save_json(q, 'data/m1/%s.json' % q['ID'])

            t.set_description("recall: %.2f" % (avg_recall / total))
            t.refresh()  # to show immediately the update
            t.update(1)
        except Exception as e:
            print(e)
            print(qs['ID'])
    return qs
    # save_json(qs, 'datasets/complexwebq/all_questions_v3_hop2_input.json')


if __name__ == '__main__':
    # process_vocab('datasets/complexwebq/glove.6B.100d.txt')
    # process_entities()
    # rel_indexer = {}
    # with open('datasets/complexwebq/relations.txt') as f:
    #     lines = f.readlines()
    #     for i, line in enumerate(lines):
    #         line = line.strip()
    #         rel_indexer[line] = i
    # questions = load_json(QUESTION_FILE)
    #
    # q_emb = question_embedding(questions)
    #
    # r_emb = relation_embeddings(questions)
    #
    # _m1 = partial(m1, q_emb, r_emb, rel_indexer)
    #
    # with Pool(processes=PARALLEL_PROCESSOR) as pool:
    #     res = pool.map(_m1,
    #                    [questions[i * len(questions) // PARALLEL_PROCESSOR:
    #                               (i + 1) * len(questions) // PARALLEL_PROCESSOR] for i in range(PARALLEL_PROCESSOR)])
    #
    # final_questions = [item for sublist in res for item in sublist]
    # save_json(final_questions, 'datasets/complexwebq/questions/all_questions_v3_hop4_input.json')
    relation_emb_100d = np.load('datasets/complexwebq/relation_emb_100d.npy')
    bi_direction_relation_emb_100d = np.concatenate((relation_emb_100d, relation_emb_100d), axis=0)
    np.save('datasets/complexwebq/birelation_emb_100d.npy', bi_direction_relation_emb_100d)