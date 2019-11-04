import sys

from pullnet_data_loader import DataLoader
from util import *

FACT_FILES = ['datasets/complexwebq/first_hop_facts.json',
              'datasets/complexwebq/second_hop_facts.json',
              'datasets/complexwebq/third_hop_facts.json']
QUESTION_FILE = 'datasets/complexwebq/questions/new_questions.json'



class PullNet(object):
    def __init__(self, fact_files):
        facts = [load_json(e) for e in fact_files]
        self.facts = facts


def train(cfg):
    T = 3
    questions = load_json(QUESTION_FILE)
    facts = [load_json(e) for e in FACT_FILES]
    facts = {**facts[0], **facts[1], **facts[2]}
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])
    for t in range(1, T + 1):
        train_documents, train_document_entity_indices, train_document_texts = None, None, None
        train_data = DataLoader(questions, facts, t, train_documents, train_document_entity_indices, train_document_texts, word2id, relation2id, cfg['max_query_word'], cfg['max_document_word'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation'])



def test(cfg):
    pass


if __name__ == "__main__":
    config_file = sys.argv[2]
    CFG = get_config(config_file)
    if '--train' == sys.argv[1]:
        train(CFG)
    elif '--test' == sys.argv[1]:
        test(CFG)
    else:
        assert False, "--train or --test?"