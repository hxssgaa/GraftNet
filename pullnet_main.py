import sys
import time

from pullnet import PullNet
from graftnet import GraftNet
from pullnet_data_loader import DataLoader
from fpnet_data_loader import FpNetDataLoader
from relreasoner import RelReasoner
from relreasoner_data_loader import RelReasonerDataLoader
from fpnet import FactsPullNet
from util import *


import warnings
warnings.filterwarnings("ignore")


T = 3


def train(cfg):
    facts = load_fact(cfg['fact_data'])
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    id2entity = {idx: entity for entity, idx in entity2id.items()}

    train_documents, train_document_entity_indices, train_document_texts = None, None, None
    train_data = DataLoader(cfg['data_folder'] + cfg['train_data'], train_documents, train_document_entity_indices,
                            train_document_texts, word2id, relation2id, cfg['max_query_word'],
                            cfg['max_document_word'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation'])

    valid_documents, valid_document_entity_indices, valid_document_texts = None, None, None
    valid_data = DataLoader(cfg['data_folder'] + cfg['dev_data'], valid_documents, valid_document_entity_indices,
                            valid_document_texts, word2id, relation2id, cfg['max_query_word'],
                            cfg['max_document_word'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation'])

    # create model & set parameters
    my_model = get_model(facts, entity2id, relation2id, cfg, valid_data.num_kb_relation, len(entity2id), len(word2id), T)
    trainable_parameters = [p for p in my_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg['learning_rate'])

    best_dev_acc = 0.0

    for epoch in range(cfg['num_epoch']):
        try:
            print('epoch', epoch)
            train_data.reset_batches(is_sequential=cfg['is_debug'])
            # Train
            my_model.train()
            my_model.teacher_force = True
            train_loss, train_hit_at_one, train_precision, train_recall, train_f1, train_max_acc, train_facts_recall = [], [], [], [], [], [], []
            for iteration in tqdm(range(train_data.num_data // cfg['batch_size'])):
                batch = train_data.get_batch(iteration, cfg['batch_size'], cfg['fact_dropout'])

                # Perform expend process, we need to initialize the pulled nodes to be topic entities first.
                question_data = batch[-1]
                for e in question_data:
                    e['hops'] = [e['entities']]
                    e['subgraph'] = {}
                pred = None
                answer_dist = None
                facts_recall = None
                loss = None
                for cur_hop in range(T):
                    if cur_hop < T - 1:
                        with torch.no_grad():
                            _, pred, pred_dist, global2local_entity_maps, answer_dist, facts_recall = my_model(batch, cur_hop + 1)
                    else:
                        loss, pred, pred_dist, global2local_entity_maps, answer_dist, facts_recall = my_model(batch,
                                                                                                              cur_hop + 1)

                    # Pull entities which could expend to the next hop
                    for i, p_dist in enumerate(pred_dist):
                        l2g = {l: g for g, l in global2local_entity_maps[i].items()}
                        output_dist = list(
                            {id2entity[l2g[j]] for j, prob in enumerate(p_dist.data.cpu().numpy()) if
                             j < len(l2g)})
                        question_data[i]['hops'].append(output_dist)

                pred = pred.data.cpu().numpy()
                hit_at_one, precision, recall, f1, max_acc = cal_accuracy(pred, answer_dist)
                train_loss.append(loss.item())
                train_hit_at_one.append(hit_at_one)
                train_precision.append(precision)
                train_recall.append(recall)
                train_f1.append(f1)
                train_max_acc.append(max_acc)
                train_facts_recall.append(facts_recall)
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
            print('avg_training_facts_recall', sum(train_facts_recall) / len(train_facts_recall))
            print('avg_training_f1', sum(train_f1) / len(train_f1))

            print("validating ...")
            eval_acc = inference(my_model, valid_data, entity2id, cfg)
            if eval_acc > best_dev_acc and cfg['to_save_model']:
                print("saving model to", cfg['save_model_file'])
                torch.save(my_model.state_dict(), cfg['save_model_file'])
                best_dev_acc = eval_acc

        except KeyboardInterrupt:
            break


def train_pullfacts(cfg):
    facts = load_fact2(cfg['fact_data'])
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])

    train_data = FpNetDataLoader(cfg['data_folder'] + cfg['train_data'], facts,
                            entity2id, word2id, relation2id, cfg['max_query_word'], cfg['use_inverse_relation'], teacher_force=False)

    valid_data = FpNetDataLoader(cfg['data_folder'] + cfg['dev_data'], facts,
                            entity2id, word2id, relation2id, cfg['max_query_word'], cfg['use_inverse_relation'])

    # create model & set parameters
    my_model = get_fpnet_model(cfg, valid_data.num_kb_relation, len(entity2id), len(word2id))
    trainable_parameters = [p for p in my_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg['learning_rate'])

    best_dev_recall = 0.0

    for epoch in range(cfg['num_epoch']):
        try:
            print('epoch', epoch)
            train_data.reset_batches(is_sequential=cfg['is_debug'])
            # Train
            my_model.train()
            train_loss, train_hit_at_one, train_precision, train_recall, train_f1, train_max_acc = [], [], [], [], [], []
            for iteration in tqdm(range(train_data.num_data // cfg['batch_size'])):
                batch = train_data.get_batch(iteration, cfg['batch_size'], cfg['fact_dropout'])
                loss, pred, _ = my_model(batch)
                #pred = pred.data.cpu().numpy()
                hit_at_one, _, recall, _, max_acc = cal_accuracy(pred, batch[-1])
                train_hit_at_one.append(hit_at_one)
                train_loss.append(loss.item())
                train_recall.append(recall)
                train_max_acc.append(max_acc)
                # back propogate
                my_model.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(my_model.parameters(), cfg['gradient_clip'])
                optimizer.step()
            print('avg_hit_at_one', sum(train_hit_at_one) / len(train_hit_at_one))
            print('avg_training_loss', sum(train_loss) / len(train_loss))
            print('max_training_acc', sum(train_max_acc) / len(train_max_acc))
            print('avg_training_recall', sum(train_recall) / len(train_recall))

            print("validating ...")
            eval_recall = inference_fpnet(my_model, valid_data, entity2id, cfg)
            if eval_recall > best_dev_recall and cfg['save_fpnet_model_file']:
                print("saving model to", cfg['save_fpnet_model_file'])
                torch.save(my_model.state_dict(), cfg['save_fpnet_model_file'])
                best_dev_recall = eval_recall

        except KeyboardInterrupt:
            break


def inference_relreasoner(my_model, data, entity2id, cfg, log_info=False):
    # Evaluation
    my_model.eval()
    my_model.teacher_force = False
    eval_hit_at_one, eval_loss, eval_recall, eval_max_acc = [], [], [], []
    id2entity = {idx: entity for entity, idx in entity2id.items()}
    data.reset_batches(is_sequential = True)
    test_batch_size = 20
    if log_info:
        f_pred = open(cfg['pred_file'], 'w')
    for iteration in tqdm(range(data.num_data // test_batch_size)):
        batch = data.get_batch(iteration, test_batch_size, fact_dropout=0.0)
        loss, pred, _ = my_model(batch)
        pred = pred.data.cpu().numpy()
        local_kb_rel_path_rels = data.local_kb_rel_path_rels[iteration * test_batch_size: (iteration + 1) * test_batch_size]
        hit_at_one, _, recall, _, max_acc = cal_accuracy(pred, batch[-1])
        eval_hit_at_one.append(hit_at_one)
        eval_loss.append(loss.item())
        eval_recall.append(recall)
        eval_max_acc.append(max_acc)

    print('avg_loss', sum(eval_loss) / len(eval_loss))
    print('max_acc', sum(eval_max_acc) / len(eval_max_acc))
    print('avg_hit_at_one', sum(eval_hit_at_one) / len(eval_hit_at_one))
    print('avg_recall', sum(eval_recall) / len(eval_recall))

    return sum(eval_recall) / len(eval_recall)


def train_relreasoner(cfg):
    facts = load_fact2(cfg['fact_data'])
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    num_hop = 3

    train_data = RelReasonerDataLoader(cfg['data_folder'] + cfg['train_data'], facts, num_hop,
                                       word2id, relation2id, cfg['max_query_word'], cfg['use_inverse_relation'], 1)

    valid_data = RelReasonerDataLoader(cfg['data_folder'] + cfg['dev_data'], facts, num_hop,
                                       word2id, relation2id, cfg['max_query_word'], cfg['use_inverse_relation'], 1)

    my_model = get_relreasoner_model(cfg, num_hop, valid_data.num_kb_relation, len(entity2id), len(word2id))
    trainable_parameters = [p for p in my_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=1e-4)

    best_dev_recall = 0.0

    for epoch in range(cfg['num_epoch']):
        try:
            print('epoch', epoch)
            train_data.reset_batches(is_sequential=cfg['is_debug'])
            # Train
            my_model.train()
            train_loss, train_hit_at_one, train_precision, train_recall, train_f1, train_max_acc = [], [], [], [], [], []
            for iteration in tqdm(range(train_data.num_data // cfg['batch_size'])):
                batch = train_data.get_batch(iteration, cfg['batch_size'], cfg['fact_dropout'])
                loss, pred, _ = my_model(batch)
                # pred = pred.data.cpu().numpy()
                hit_at_one, _, recall, _, max_acc = cal_accuracy(pred, batch[-1])
                train_hit_at_one.append(hit_at_one)
                train_loss.append(loss.item())
                train_recall.append(recall)
                train_max_acc.append(max_acc)
                # back propogate
                my_model.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(my_model.parameters(), cfg['gradient_clip'])
                optimizer.step()
            print('avg_hit_at_one', sum(train_hit_at_one) / len(train_hit_at_one))
            print('avg_training_loss', sum(train_loss) / len(train_loss))
            print('max_training_acc', sum(train_max_acc) / len(train_max_acc))
            print('avg_training_recall', sum(train_recall) / len(train_recall))

            print("validating ...")
            eval_recall = inference_relreasoner(my_model, valid_data, entity2id, cfg)
            if eval_recall > best_dev_recall and cfg['save_fpnet_model_file']:
                print("saving model to", cfg['save_fpnet_model_file'])
                torch.save(my_model.state_dict(), cfg['save_fpnet_model_file'])
                best_dev_recall = eval_recall

        except KeyboardInterrupt:
            break


def prediction_relreasoner(cfg):
    facts = load_fact2(cfg['fact_data'])
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    num_hop = 3

    # train_data = RelReasonerDataLoader(cfg['data_folder'] + cfg['train_data'], facts, num_hop,
    #                                    word2id, relation2id, cfg['max_query_word'], cfg['use_inverse_relation'], 1)

    # valid_data = RelReasonerDataLoader(cfg['data_folder'] + cfg['dev_data'], facts, num_hop,
    #                                    word2id, relation2id, cfg['max_query_word'], cfg['use_inverse_relation'], 1)

    test_data = RelReasonerDataLoader(cfg['data_folder'] + cfg['test_data'], facts, num_hop,
                                       word2id, relation2id, cfg['max_query_word'], cfg['use_inverse_relation'], 1)

    my_model = get_relreasoner_model(cfg, num_hop, test_data.num_kb_relation, len(entity2id), len(word2id))

    eval_recall = inference_relreasoner(my_model, test_data, entity2id, cfg)



def test(cfg):
    pass


def get_model(facts, entity2id, relation2id, cfg, num_kb_relation, num_entities, num_vocab, num_iteration):
    word_emb_file = None if cfg['word_emb_file'] is None else cfg['data_folder'] + cfg['word_emb_file']
    entity_emb_file = None if cfg['entity_emb_file'] is None else cfg['data_folder'] + cfg['entity_emb_file']
    entity_kge_file = None if cfg['entity_kge_file'] is None else cfg['data_folder'] + cfg['entity_kge_file']
    relation_emb_file = None if cfg['relation_emb_file'] is None else cfg['data_folder'] + cfg['relation_emb_file']
    relation_kge_file = None if cfg['relation_kge_file'] is None else cfg['data_folder'] + cfg['relation_kge_file']

    my_model = use_cuda(PullNet(facts, entity2id, relation2id, word_emb_file, entity_emb_file, entity_kge_file, relation_emb_file, relation_kge_file,
                                 cfg['num_layer'], num_kb_relation, num_entities, num_vocab, cfg['entity_dim'],
                                 cfg['word_dim'], cfg['kge_dim'], cfg['pagerank_lambda'], cfg['fact_scale'],
                                 cfg['lstm_dropout'], cfg['linear_dropout'], cfg['fact_dropout'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation'], num_iteration, False))

    if cfg['load_model_file'] is not None:
        print('loading model from', cfg['load_model_file'])
        pretrained_model_states = torch.load(cfg['load_model_file'])
        if word_emb_file is not None:
            del pretrained_model_states['word_embedding.weight']
        if entity_emb_file is not None:
            del pretrained_model_states['entity_embedding.weight']
        my_model.load_state_dict(pretrained_model_states, strict=False)

    return my_model


def get_relreasoner_model(cfg, num_hop, num_kb_relation, num_entities, num_vocab):
    word_emb_file = None if cfg['word_emb_file'] is None else cfg['data_folder'] + cfg['word_emb_file']
    relation_emb_file = None if cfg['relation_emb_file'] is None else cfg['data_folder'] + cfg['relation_emb_file']

    my_model = use_cuda(RelReasoner(word_emb_file, relation_emb_file,
                                     num_kb_relation, num_entities, num_vocab, num_hop, cfg['entity_dim'],
                                     cfg['word_dim'], cfg['lstm_dropout'], cfg['use_inverse_relation']))

    if cfg['load_fpnet_model_file'] is not None:
        print('loading model from', cfg['load_fpnet_model_file'])
        pretrained_model_states = torch.load(cfg['load_fpnet_model_file'])
        if word_emb_file is not None:
            del pretrained_model_states['word_embedding.weight']
        my_model.load_state_dict(pretrained_model_states, strict=False)

    return my_model


def get_fpnet_model(cfg, num_kb_relation, num_entities, num_vocab):
    word_emb_file = None if cfg['word_emb_file'] is None else cfg['data_folder'] + cfg['word_emb_file']
    relation_emb_file = None if cfg['relation_emb_file'] is None else cfg['data_folder'] + cfg['relation_emb_file']

    my_model = use_cuda(FactsPullNet(word_emb_file, relation_emb_file,
                                     num_kb_relation, num_entities, num_vocab, cfg['entity_dim'],
                                     cfg['word_dim'], cfg['lstm_dropout'], cfg['use_inverse_relation']))

    if cfg['load_fpnet_model_file'] is not None:
        print('loading model from', cfg['load_fpnet_model_file'])
        pretrained_model_states = torch.load(cfg['load_fpnet_model_file'])
        if word_emb_file is not None:
            del pretrained_model_states['word_embedding.weight']
        my_model.load_state_dict(pretrained_model_states, strict=False)

    return my_model


def get_graftnet_model(cfg, num_kb_relation, num_entities, num_vocab):
    word_emb_file = None if cfg['word_emb_file'] is None else cfg['data_folder'] + cfg['word_emb_file']
    entity_emb_file = None if cfg['entity_emb_file'] is None else cfg['data_folder'] + cfg['entity_emb_file']
    entity_kge_file = None if cfg['entity_kge_file'] is None else cfg['data_folder'] + cfg['entity_kge_file']
    relation_emb_file = None if cfg['relation_emb_file'] is None else cfg['data_folder'] + cfg['relation_emb_file']
    relation_kge_file = None if cfg['relation_kge_file'] is None else cfg['data_folder'] + cfg['relation_kge_file']

    my_model = use_cuda(GraftNet(word_emb_file, entity_emb_file, entity_kge_file, relation_emb_file, relation_kge_file,
                                 1, num_kb_relation, num_entities, num_vocab, cfg['entity_dim'],
                                 cfg['word_dim'], cfg['kge_dim'], cfg['pagerank_lambda'], cfg['fact_scale'],
                                 cfg['lstm_dropout'], cfg['linear_dropout'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation']))

    if cfg['load_fpnet_model_file'] is not None:
        print('loading model from', cfg['load_fpnet_model_file'])
        pretrained_model_states = torch.load(cfg['load_fpnet_model_file'])
        if word_emb_file is not None:
            del pretrained_model_states['word_embedding.weight']
        if entity_emb_file is not None:
            del pretrained_model_states['entity_embedding.weight']
        my_model.load_state_dict(pretrained_model_states, strict=False)

    return my_model


def inference(my_model, valid_data, entity2id, cfg, log_info=False):
    # Evaluation
    my_model.eval()
    my_model.teacher_force = False
    eval_loss, eval_hit_at_one, eval_precision, eval_recall, eval_f1, eval_max_acc, eval_facts_recall = [], [], [], [], [], [], []
    id2entity = {idx: entity for entity, idx in entity2id.items()}
    valid_data.reset_batches(is_sequential = True)
    test_batch_size = 20
    if log_info:
        f_pred = open(cfg['pred_file'], 'w')
    for iteration in tqdm(range(valid_data.num_data // test_batch_size)):
        batch = valid_data.get_batch(iteration, test_batch_size, fact_dropout=0.0)
        question_data = batch[-1]
        for e in question_data:
            e['hops'] = [e['entities']]
            e['subgraph'] = {}
        for cur_hop in range(T):
            loss, pred, pred_dist, global2local_entity_maps, answer_dist, facts_recall = my_model(batch, cur_hop + 1)

            # Pull entities which could expend to the next hop
            for i, p_dist in enumerate(pred_dist):
                l2g = {l: g for g, l in global2local_entity_maps[i].items()}
                output_dist = list(
                    {id2entity[l2g[j]] for j, prob in enumerate(p_dist.data.cpu().numpy()) if
                     j < len(l2g)})
                question_data[i]['hops'].append(output_dist)

            if cur_hop == T - 1:
                pred = pred.data.cpu().numpy()
                hit_at_one, precision, recall, f1, max_acc = cal_accuracy(pred, answer_dist)
                if log_info:
                    output_pred_dist(pred_dist, batch[-1], id2entity, iteration * test_batch_size, valid_data, f_pred)
                eval_loss.append(loss.item())
                eval_hit_at_one.append(hit_at_one)
                eval_precision.append(precision)
                eval_recall.append(recall)
                eval_facts_recall.append(facts_recall)
                eval_f1.append(f1)
                eval_max_acc.append(max_acc)

    print('avg_loss', sum(eval_loss) / len(eval_loss))
    print('max_acc', sum(eval_max_acc) / len(eval_max_acc))
    print('avg_hit@1', sum(eval_hit_at_one) / len(eval_hit_at_one))
    print('avg_precision', sum(eval_precision) / len(eval_precision))
    print('avg_recall', sum(eval_recall) / len(eval_recall))
    print('avg_facts_recall', sum(eval_facts_recall) / len(eval_facts_recall))
    print('avg_f1', sum(eval_f1) / len(eval_f1))

    return sum(eval_precision) / len(eval_precision)


def inference_fpnet(my_model, valid_data, entity2id, cfg, log_info=False):
    # Evaluation
    my_model.eval()
    my_model.teacher_force = False
    eval_hit_at_one, eval_loss, eval_recall, eval_max_acc = [], [], [], []
    id2entity = {idx: entity for entity, idx in entity2id.items()}
    valid_data.reset_batches(is_sequential = True)
    test_batch_size = 20
    if log_info:
        f_pred = open(cfg['pred_file'], 'w')
    for iteration in tqdm(range(valid_data.num_data // test_batch_size)):
        batch = valid_data.get_batch(iteration, test_batch_size, fact_dropout=0.0)
        loss, pred, _ = my_model(batch)
        pred = pred.data.cpu().numpy()
        hit_at_one, _, recall, _, max_acc = cal_accuracy(pred, batch[-1])
        #if log_info:
        #    output_pred_dist(pred_dist, batch[-1], id2entity, iteration * test_batch_size, valid_data, f_pred)
        eval_hit_at_one.append(hit_at_one)
        eval_loss.append(loss.item())
        eval_recall.append(recall)
        eval_max_acc.append(max_acc)

    print('avg_loss', sum(eval_loss) / len(eval_loss))
    print('max_acc', sum(eval_max_acc) / len(eval_max_acc))
    print('avg_hit_at_one', sum(eval_hit_at_one) / len(eval_hit_at_one))
    print('avg_recall', sum(eval_recall) / len(eval_recall))

    return sum(eval_recall) / len(eval_recall)



if __name__ == "__main__":
    config_file = sys.argv[2]
    CFG = get_config(config_file)
    if '--train' == sys.argv[1]:
        train(CFG)
    elif '--test' == sys.argv[1]:
        test(CFG)
    elif '--train-pullfacts' == sys.argv[1]:
        train_pullfacts(CFG)
    elif '--train-relreasoner' == sys.argv[1]:
        train_relreasoner(CFG)
    elif '--prediction-relreasoner' == sys.argv[1]:
        prediction_relreasoner(CFG)
    else:
        assert False, "--train or --test?"