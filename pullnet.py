import sys

from pullnet_data_loader import DataLoader
from util import *
from graftnet import GraftNet

FACT_FILE = 'datasets/complexwebq/all_facts.json'
QUESTION_FILE = 'datasets/complexwebq/questions/all_questions_v3_hop3_input.json'
OUT_PREDICTION_QUESTION_FILE = 'datasets/complexwebq/questions/all_questions_v3_hop%d.json'
RAW_QUESTION_IDS = {
    'train': set(map(lambda x: x['ID'], load_json('datasets/complexwebq/questions/ComplexWebQuestions_train.json'))),
    'dev': set(map(lambda x: x['ID'], load_json('datasets/complexwebq/questions/ComplexWebQuestions_dev.json'))),
    'test': set(map(lambda x: x['ID'], load_json('datasets/complexwebq/questions/ComplexWebQuestions_test.json'))),
}


def train(cfg, is_test=False):
    T = 1
    questions = load_json(QUESTION_FILE)
    facts = None
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])

    train_questions = [q for q in questions if q['ID'] in RAW_QUESTION_IDS['train']]
    dev_questions = [q for q in questions if q['ID'] in RAW_QUESTION_IDS['dev']]
    test_questions = [q for q in questions if q['ID'] in RAW_QUESTION_IDS['test']]
    trainable_entities = set()
    for q in train_questions:
        for t in q['subgraph']['tuples']:
            s, p, o = t
            trainable_entities.add(entity2id[s])
            trainable_entities.add(entity2id[o])

    t = 4
    train_documents, train_document_entity_indices, train_document_texts = None, None, None
    train_data = DataLoader(train_questions, facts, t, train_documents, train_document_entity_indices, train_document_texts, word2id, relation2id, entity2id, cfg['max_query_word'], cfg['max_document_word'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation'])

    dev_documents, dev_document_entity_indices, dev_document_texts = None, None, None
    valid_data = DataLoader(dev_questions, facts, t, dev_documents, dev_document_entity_indices, dev_document_texts,
                            word2id, relation2id, entity2id, cfg['max_query_word'],
                            cfg['max_document_word'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation'])

    test_documents, test_document_entity_indices, test_document_texts = None, None, None
    test_data = DataLoader(test_questions, facts, t, test_documents, test_document_entity_indices, test_document_texts,
                            word2id, relation2id, entity2id, cfg['max_query_word'],
                            cfg['max_document_word'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation'])

    # create model & set parameters
    my_model = get_model(trainable_entities, facts, entity2id, relation2id, cfg, train_data.num_kb_relation, len(entity2id),
                         len(word2id))

    trainable_parameters = [p for p in my_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg['learning_rate'])

    best_dev_acc = 0.0
    for epoch in range(cfg['num_epoch']):
        try:
            print('epoch', epoch)
            if not is_test:
                train_data.reset_batches(is_sequential=cfg['is_debug'])
                # Train
                my_model.train()
                train_loss, train_hit_at_one, train_precision, train_recall, train_f1, train_max_acc = [], [], [], [], [], []
                for iteration in tqdm(range(train_data.num_data // cfg['batch_size'])):
                    batch = train_data.get_batch(iteration, cfg['batch_size'], cfg['fact_dropout'])
                    loss, pred, _, = my_model(batch)
                    pred = pred.data.cpu().numpy()
                    hit_at_one, precision, recall, f1, max_acc = cal_accuracy(pred, batch[-1])
                    train_loss.append(loss.item())
                    train_hit_at_one.append(hit_at_one)
                    train_precision.append(precision)
                    train_recall.append(recall)
                    train_f1.append(f1)
                    train_max_acc.append(max_acc)
                    # back propogate
                    my_model.zero_grad()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(my_model.parameters(), cfg['gradient_clip'])
                    optimizer.step()
                print('avg_training_loss', sum(train_loss) / len(train_loss))
                print('max_training_acc', sum(train_max_acc) / len(train_max_acc))
                print('avg_training_hit@1', sum(train_hit_at_one) / len(train_hit_at_one))
                print('avg_training_precision', sum(train_precision) / len(train_precision))
                print('avg_training_recall', sum(train_recall) / len(train_recall))
                print('avg_training_f1', sum(train_f1) / len(train_f1))

                print("validating ...")
                eval_acc = inference(my_model, valid_data, entity2id, cfg)
                if eval_acc > best_dev_acc and cfg['to_save_model']:
                    print("saving model to", cfg['save_model_file'])
                    torch.save(my_model.state_dict(), cfg['save_model_file'])
                    best_dev_acc = eval_acc

                if is_test:
                    print('testing...')
                    test_acc = inference(my_model, test_data, entity2id, cfg)

        except KeyboardInterrupt:
            break


def test(cfg):
    T = 3
    questions = load_json(QUESTION_FILE)
    facts = load_json(FACT_FILE)
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])

    test_questions = [q for q in questions if q['ID'] in RAW_QUESTION_IDS['test']]
    trainable_entities = set()

    #TODO: Support multi hop prediction.
    dev_documents, dev_document_entity_indices, dev_document_texts = None, None, None
    test_data = DataLoader(trainable_entities, facts, test_questions, facts, 1, dev_documents, dev_document_entity_indices, dev_document_texts,
                            word2id, relation2id, entity2id, cfg['max_query_word'],
                            cfg['max_document_word'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation'])

    my_model = get_model(cfg, test_data.num_kb_relation, len(entity2id), len(word2id))
    test_acc = inference(my_model, test_data, entity2id, cfg, log_info=True)


def prediction(cfg, t):
    questions = load_json(QUESTION_FILE)

    test_questions = [q for q in questions if q['ID'] in RAW_QUESTION_IDS['test']]
    facts = None
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])

    trainable_entities = set()

    documents, document_entity_indices, document_texts = None, None, None
    data = DataLoader(test_questions, facts, t, documents, document_entity_indices,
                            document_texts, word2id, relation2id, entity2id, cfg['max_query_word'],
                            cfg['max_document_word'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation'])


    # create model & set parameters
    my_model = get_model(trainable_entities, facts, entity2id, relation2id, cfg, data.num_kb_relation,
                         len(entity2id),
                         len(word2id))
    # Evaluation
    my_model.eval()
    eval_loss, eval_hit_at_one, eval_precision, eval_recall, eval_f1, eval_max_acc = [], [], [], [], [], []
    id2entity = {idx: entity for entity, idx in entity2id.items()}
    data.reset_batches(is_sequential=True)
    test_batch_size = 20
    for iteration in tqdm(range(data.num_data // test_batch_size)):
        batch = data.get_batch(iteration, test_batch_size, fact_dropout=0.0)
        loss, pred, pred_dist = my_model(batch)
        pred = pred.data.cpu().numpy()
        hit_at_one, precision, recall, f1, max_acc = cal_accuracy(pred, batch[-1])
        pred_list = output_pred_dist(pred, batch[-1], id2entity, iteration * test_batch_size, data, None)
        for idx in range(len(pred_list)):
            questions[iteration * test_batch_size + idx]['hop_%d' % t] = pred_list[idx]
        eval_loss.append(loss.item())
        eval_hit_at_one.append(hit_at_one)
        eval_precision.append(precision)
        eval_recall.append(recall)
        eval_f1.append(f1)
        eval_max_acc.append(max_acc)

    print('avg_loss', sum(eval_loss) / len(eval_loss))
    print('max_acc', sum(eval_max_acc) / len(eval_max_acc))
    print('avg_hit@1', sum(eval_hit_at_one) / len(eval_hit_at_one))
    print('avg_precision', sum(eval_precision) / len(eval_precision))
    print('avg_recall', sum(eval_recall) / len(eval_recall))
    print('avg_f1', sum(eval_f1) / len(eval_f1))

    # save_json(questions, OUT_PREDICTION_QUESTION_FILE % t)


def get_model(trainable_entities, facts, entity2id, relation2id, cfg, num_kb_relation, num_entities, num_vocab):
    word_emb_file = None if cfg['word_emb_file'] is None else cfg['data_folder'] + cfg['word_emb_file']
    entity_emb_file = None if cfg['entity_emb_file'] is None else cfg['data_folder'] + cfg['entity_emb_file']
    entity_kge_file = None if cfg['entity_kge_file'] is None else cfg['data_folder'] + cfg['entity_kge_file']
    relation_emb_file = None if cfg['relation_emb_file'] is None else cfg['data_folder'] + cfg['relation_emb_file']
    relation_kge_file = None if cfg['relation_kge_file'] is None else cfg['data_folder'] + cfg['relation_kge_file']

    my_model = use_cuda(GraftNet(trainable_entities, word_emb_file, entity_emb_file, entity_kge_file, relation_emb_file, relation_kge_file,
                                 cfg['num_layer'], num_kb_relation, num_entities, num_vocab, cfg['entity_dim'],
                                 cfg['word_dim'], cfg['kge_dim'], cfg['pagerank_lambda'], cfg['fact_scale'],
                                 cfg['lstm_dropout'], cfg['linear_dropout'], cfg['use_kb'], cfg['use_doc']))

    if cfg['load_model_file'] is not None:
        print('loading model from', cfg['load_model_file'])
        pretrained_model_states = torch.load(cfg['load_model_file'])
        if word_emb_file is not None:
            del pretrained_model_states['word_embedding.weight']
        if entity_emb_file is not None:
            del pretrained_model_states['entity_embedding.weight']
        my_model.load_state_dict(pretrained_model_states, strict=False)

    return my_model


def inference(my_model, valid_data, entity2id, cfg, log_info=False):
    # Evaluation
    my_model.eval()
    eval_loss, eval_hit_at_one, eval_precision, eval_recall, eval_f1, eval_max_acc = [], [], [], [], [], []
    id2entity = {idx: entity for entity, idx in entity2id.items()}
    valid_data.reset_batches(is_sequential = True)
    test_batch_size = 20
    if log_info:
        f_pred = open(cfg['pred_file'], 'w')
    for iteration in tqdm(range(valid_data.num_data // test_batch_size)):
        batch = valid_data.get_batch(iteration, test_batch_size, fact_dropout=0.0)
        loss, pred, pred_dist = my_model(batch)
        pred = pred.data.cpu().numpy()
        hit_at_one, precision, recall, f1, max_acc = cal_accuracy(pred, batch[-1])
        if log_info:
            output_pred_dist(pred_dist, batch[-1], id2entity, iteration * test_batch_size, valid_data, f_pred)
        eval_loss.append(loss.item())
        eval_hit_at_one.append(hit_at_one)
        eval_precision.append(precision)
        eval_recall.append(recall)
        eval_f1.append(f1)
        eval_max_acc.append(max_acc)

    print('avg_loss', sum(eval_loss) / len(eval_loss))
    print('max_acc', sum(eval_max_acc) / len(eval_max_acc))
    print('avg_hit@1', sum(eval_hit_at_one) / len(eval_hit_at_one))
    print('avg_precision', sum(eval_precision) / len(eval_precision))
    print('avg_recall', sum(eval_recall) / len(eval_recall))
    print('avg_f1', sum(eval_f1) / len(eval_f1))

    return sum(eval_hit_at_one) / len(eval_hit_at_one)


if __name__ == "__main__":
    config_file = sys.argv[2]
    CFG = get_config(config_file)
    if '--train' == sys.argv[1]:
        train(CFG)
    elif '--test' == sys.argv[1]:
        train(CFG, is_test=True)
    elif '--prediction' == sys.argv[1]:
        prediction(CFG, 3)
    else:
        assert False, "--train or --test?"