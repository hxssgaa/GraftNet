import json
import os
import numpy as np


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
    entities.update(load_json('datasets/complexwebq/ground_truth_entities.json'))
    entities = list(sorted(entities))
    with open('datasets/complexwebq/entities.txt', 'w') as f:
        for e in entities:
            if e.startswith('m.') or e.startswith('g.'):
                f.writelines(e + '\n')


def process_vocab():
    vocab2id = load_json('datasets/complexwebq/vocab2id.json')
    with open('datasets/complexwebq/vocab.txt', 'w') as f:
        for e in vocab2id:
            f.writelines(e + '\n')


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


if __name__ == '__main__':
    process_entities()
