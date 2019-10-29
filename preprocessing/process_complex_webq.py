import json
import os
from string import punctuation

import numpy as np
import wordninja
from tqdm import tqdm


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
    word_emb_npy = np.array([word2vec[word] if word in word2vec else np.random.uniform(-1, 1, 100) for word in corpus_list])
    with open('datasets/complexwebq/vocab.txt', 'w') as f:
        for e in corpus_list:
            f.writelines(e + '\n')
    np.save('datasets/complexwebq/word_emb_100d.npy', word_emb_npy)
    print('train emb coverage: ', len(corpus & set(word2vec.keys())) / len(corpus))
    print('dev coverage: ', len(corpus & dev_corpus) / len(dev_corpus))
    print('test coverage: ', len(corpus & test_corpus) / len(test_corpus))


if __name__ == '__main__':
    # process_vocab('datasets/complexwebq/glove.6B.100d.txt')
    process_entities()