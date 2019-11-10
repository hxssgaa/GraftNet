import json
import os
from string import punctuation

import numpy as np
import wordninja
import pickle as pkl
from tqdm import tqdm
from preprocessing.use_helper import UseVector

RELATIONS_FILE = 'datasets/complexwebq/relations.txt'
OUT_RELATIONS_EMBEDDING = 'datasets/complexwebq/relation_emb.pkl'
QUESTION_FILE = 'datasets/complexwebq/questions/all_questions.json'
WORD_DIM = 200


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
    facts = load_json('datasets/complexwebq/all_facts.json')
    result = set()
    for k1, v1 in tqdm(facts.items()):
        for k2, v2 in v1.items():
            _, rel = v2
            result.add(rel)
    result = list(sorted(result))
    with open('datasets/complexwebq/relations.txt', 'w') as f:
        for e in result:
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
    word_emb_npy = np.array([word2vec[word] if word in word2vec else np.random.uniform(-1, 1, 200) for word in corpus_list])
    with open('datasets/complexwebq/vocab.txt', 'w') as f:
        for e in corpus_list:
            f.writelines(e + '\n')
    np.save('datasets/complexwebq/word_emb_200d.npy', word_emb_npy)
    print('train emb coverage: ', len(corpus & set(word2vec.keys())) / len(corpus))
    print('dev coverage: ', len(corpus & dev_corpus) / len(dev_corpus))
    print('test coverage: ', len(corpus & test_corpus) / len(test_corpus))


def process_relation_embedding(embedding_file, use_f=True):
    if os.path.exists(OUT_RELATIONS_EMBEDDING) and use_f:
        return pkl.load(open(OUT_RELATIONS_EMBEDDING, 'rb'))

    word_to_relation = {}
    relation_lens = {}

    def _add_word(word, t, v):
        if word not in word_to_relation:
            word_to_relation[word] = []
        word_to_relation[word].append((v, t))
        if v not in relation_lens:
            relation_lens[v] = 0
        relation_lens[v] += t

    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    with open(RELATIONS_FILE) as f:
        relations = f.readlines()
    for ii, line in enumerate(relations):
        relation = line.strip()
        if relation in ["GreaterThan", "LessThan", "NotEquals"]:
            print(relation)
            relation = 'common.' + relation + '.object'
        elif relation.count('.') < 2:
            print(relation)
            relation = 'common.notable_for.object'
        domain, typ, prop = relation.split(".")[-3:]
        for word in domain.split("_"):
            _add_word(word, 1, relation)
        for word in typ.split("_"):
            _add_word(word, 2, relation)
        for word in prop.split("_"):
            _add_word(word, 3, relation)

    relation_emb = {r: np.zeros((WORD_DIM,)) for r in relation_lens}
    with open(embedding_file) as f:
        for line in tqdm(f):
            spt_line = line.strip().split()
            for idx_e, e in enumerate(spt_line):
                if is_float(e):
                    word, vec = ' '.join(spt_line[:idx_e]), ' '.join(spt_line[idx_e:])
                    if word in word_to_relation:
                        for rid, typ in word_to_relation[word]:
                            relation_emb[rid] += typ * np.array(
                                [float(vv) for vv in vec.split()])
                    break

    for relation in relation_emb:
        relation_emb[relation] = relation_emb[relation] / relation_lens[relation]

    # Filter relations which are not included in the vocabulary at
    relation_emb = {k: v for k, v in relation_emb.items() if np.average(v) != 0}
    relation_emb_npy = np.array([relation_emb[rel.strip()] if rel.strip() in relation_emb else np.random.uniform(-1, 1, 200) for rel in relations])

    pkl.dump(relation_emb, open(OUT_RELATIONS_EMBEDDING, "wb"))
    np.save('datasets/complexwebq/relation_emb_200d.npy', relation_emb_npy)
    return relation_emb


if __name__ == '__main__':
    # result = process_relation_embedding('datasets/complexwebq/glove.6B.200d.txt', False)
    # process_vocab('datasets/complexwebq/glove.6B.100d.txt')
    # process_entities()
    # process_vocab('datasets/complexwebq/glove.6B.200d.txt')

    facts = load_json('datasets/complexwebq/all_facts.json')
    questions = load_json(QUESTION_FILE)
    for q in tqdm(questions):
        q['path_v2'] = []
        for path in q['path']:
            if len(path) % 2 == 0:
                continue
            for i in range(len(path) // 2):
                i = i * 2
                s = path[i] if isinstance(path[i], list) else [path[i]]
                o = path[i + 2]
                entities = set()
                tuples = set()
                path_info = {'entities': [], 'tuples': []}
                if i == 0:
                    s = s[0]
                    entities.add(s)
                    for e in o:
                        entities.add(e)
                        if s in facts and e in facts[s]:
                            direction, rel = facts[s][e]
                            if direction == 0:
                                tuples.add((s, rel, e))
                            else:
                                tuples.add((e, rel, s))
                else:
                    for idx in range(min(len(s), len(o))):
                        sbj = s[idx]
                        obj = o[idx]
                        entities.add(sbj)
                        entities.add(obj)
                        if sbj in facts and obj in facts[sbj]:
                            direction, rel = facts[sbj][obj]
                            if direction == 0:
                                tuples.add((sbj, rel, obj))
                            else:
                                tuples.add((obj, rel, sbj))
                path_info['entities'] = list(sorted(entities))
                path_info['tuples'] = list(sorted(tuples))
                q['path_v2'].append(path_info)
    save_json(questions, 'datasets/complexwebq/all_questions_v2.json')


    # facts = load_json('datasets/complexwebq/all_facts.json')
    # facts_v2 = {}
    # entities = set()
    # for k1, v1 in tqdm(facts.items()):
    #     entities.add(k1)
    #     if k1 not in facts_v2:
    #         facts_v2[k1] = dict()
    #     for k2, v2 in v1.items():
    #         entities.add(k2)
    #         direction, rel = v2
    #         if rel not in facts_v2[k1]:
    #             facts_v2[k1][rel] = dict()
    #         if k2 not in facts_v2[k1][rel]:
    #             facts_v2[k1][rel][k2] = direction
    #
    # with open('datasets/complexwebq/entities_v2.txt', 'w') as f:
    #     for e in list(sorted(entities)):
    #         if e.startswith('m.') or e.startswith('g.'):
    #             f.writelines(e + '\n')
    # save_json(facts_v2, 'datasets/complexwebq/all_facts_v2.json')