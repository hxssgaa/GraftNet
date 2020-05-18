import sys
import time

from pullnet import PullNet
from graftnet import GraftNet
from data_loader import DataLoader
from fpnet_data_loader import FpNetDataLoader
from relreasoner import RelReasoner
from relreasoner_order import RelOrderReasoner
from relreasoner_entity import EntityRelReasoner
from relreasoner_data_loader import RelReasonerDataLoader
from relreasoner_object_data_loader import RelReasonerObjectDataLoader
from fpnet import FactsPullNet
from collections import defaultdict
from util import *
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")


T = 3


def train_answer_prediction(cfg):
    facts = None
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    id2entity = {idx: entity for entity, idx in entity2id.items()}

    train_documents, train_document_entity_indices, train_document_texts = None, None, None
    train_data = DataLoader(cfg['data_folder'] + cfg['train_data'], train_documents, train_document_entity_indices,
                            train_document_texts, word2id, relation2id, entity2id, cfg['max_query_word'],
                            cfg['max_document_word'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation'])

    valid_documents, valid_document_entity_indices, valid_document_texts = None, None, None
    valid_data = DataLoader(cfg['data_folder'] + cfg['dev_data'], valid_documents, valid_document_entity_indices,
                            valid_document_texts, word2id, relation2id, entity2id, cfg['max_query_word'],
                            cfg['max_document_word'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation'])

    trainable_entities = set()
    for q in train_data.data:
        for tup in q['subgraph']['tuples']:
            s, p, o = tuple(map(lambda x: x['text'] if isinstance(x, dict) else x, tup))
            s = s.replace('%', '')
            o = o.replace('%', '')
            if s.strip() in entity2id:
                trainable_entities.add(entity2id[s.strip()])
            if o.strip() in entity2id:
                trainable_entities.add(entity2id[o.strip()])

    # create model & set parameters
    my_model = get_model(trainable_entities, facts, entity2id, relation2id, cfg, valid_data.num_kb_relation, len(entity2id), len(word2id), T)
    trainable_parameters = [p for p in my_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg['learning_rate'])

    best_dev_acc = 0.0

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


def inference_answer_helper(entity, pred_path, facts, visited):
    if not pred_path:
        return {entity}
    if entity in visited or entity not in facts or pred_path[0] not in facts[entity]:
        return set()
    visited.add(entity)
    res = set()
    for neighbor in facts[entity][pred_path[0]]:
        res |= inference_answer_helper(neighbor, pred_path[1:], facts, visited)
    return res


def inference_answer(facts, questions):
    avg_hit_at_one = 0.0
    avg_recall = 0.0
    avg_f1 = 0.0
    for q in tqdm(questions):
        entity = q['entities'][0]['text']
        entity = entity.replace('%', '')
        pred_path = q['pred_rel_path']
        inference_answers = set(inference_answer_helper(entity, pred_path, facts, set()))
        actual_answers = set(map(lambda x: x['text'], q['answers']))
        avg_hit_at_one += (1 if len(inference_answers & actual_answers) > 0 else 0)
        precision = 0
        for pred_ans in inference_answers:
            if pred_ans in actual_answers:
                precision += 1
        precision = float(precision) / len(inference_answers) if len(inference_answers) > 0 else 0
        recall = 0
        for act_ans in actual_answers:
            if act_ans in inference_answers:
                recall += 1
        recall = float(recall) / len(actual_answers) if len(actual_answers) > 0 else 0
        f1 = 0
        if precision + recall > 0:
            f1 = 2 * recall * precision / (precision + recall)
        avg_recall += recall
        avg_f1 += f1
    return avg_hit_at_one / len(questions), avg_recall / len(questions), avg_f1 / len(questions)


def inference_relreasoner(my_model, test_batch_size, data, entity2id, relation2id, reverse_relation2id, cfg, T=3, facts=None, is_train=True, is_order=False, log_info=False, include_eod=True):
    # Evaluation
    # test_batch_size = 1
    my_model.eval()
    my_model.teacher_force = False
    eval_hit_at_one, eval_loss, eval_recall, eval_max_acc = [], [], [], []
    # id2entity = {idx: entity for entity, idx in entity2id.items()}
    data.reset_batches(is_sequential = True)
    if log_info:
        f_pred = open(cfg['pred_file'], 'w')
    origin_data = data.origin_data
    rel_mapping = dict()
    for iteration in tqdm(range(data.num_data // test_batch_size)):
        batch = data.get_batch(iteration, test_batch_size, fact_dropout=0.0)
        loss, pred, _ = my_model(batch, facts=facts, relation2id=relation2id, reverse_relation2id=reverse_relation2id, teacher_forcing_ratio=0.0)
        pred = pred.data.cpu().numpy().T.astype(np.int32)
        if not is_train and not is_order:
            for row in range(pred.shape[0]):
                sample = data.data[iteration * test_batch_size + row]
                cands = sample['rel_chain_cands']
                seed_entity = sample['seed_entity']

                relations = []
                for col in range(pred.shape[1]):
                    relations.append(reverse_relation2id[pred[row][col]])
                # data.data[iteration * test_batch_size + row]['pred_rel_path'] = relations
                if relations:
                    if 'ID' not in sample:
                        sample['ID'] = iteration * test_batch_size + row
                    if sample['ID'] not in rel_mapping:
                        rel_mapping[sample['ID']] = dict()
                    rel_mapping_dict = rel_mapping[sample['ID']]
                    rel_mapping_dict[seed_entity] = relations
                    # next_hop_entities = set()
                    # key = 'entities' if num_hop == 1 else 'entities_pred%d' % (num_hop - 1)
                    # for e in data.data[iteration * test_batch_size + row][key]:
                    #     if e in facts and top_relation[-1] in facts[e]:
                    #         next_hop_entities.update(list(facts[e][top_relation[-1]].keys()))
                    # next_hop_entities = list(next_hop_entities)[:5]
                    # next_hop_relations = set()
                    # for e in next_hop_entities:
                    #     next_hop_relations.update(list(facts[e].keys()))
                    # next_hop_relations = list(next_hop_relations)
                    # next_hop_relations = list(map(lambda x: list(top_relation) + [x], next_hop_relations))
                    # data.data[iteration * test_batch_size + row]['entities_pred%d' % (num_hop)] = next_hop_entities
                    # data.data[iteration * test_batch_size + row]['relation_pred%d' % (num_hop)] = top_relation
                    # data.data[iteration * test_batch_size + row]['rel_cands_multi_cands%d' % (num_hop + 1)] = next_hop_relations
        # if not is_train and is_order:
        #     for row in range(pred.shape[0]):
        #         max_pred = pred[row][0]
        #         pred_path = data.local_kb_rel_path_rels[iteration * test_batch_size + row][max_pred]
        #         relations = []
        #         for rel_id in pred_path:
        #             relations.append(reverse_relation2id[rel_id])
        #         data.data[iteration * test_batch_size + row]['pred_rel_path'] = relations

        hit_at_one, _, recall, _, max_acc = cal_accuracy(pred, batch[-1][1:].T.astype(np.int32))
        eval_hit_at_one.append(hit_at_one)
        eval_loss.append(loss.item())
        eval_recall.append(recall)
        eval_max_acc.append(max_acc)

    avg_recall = 0
    total = 0
    min_items = []
    max_items = []
    item_lens = []
    ACCEPT_OTHER_BRANCH_ENTITIES = 20
    block_rels = {'Equals', 'GreaterThan', 'GreaterThanOrEqual', 'LessThan', 'LessThanOrEqual', 'NotEquals'}
    err = 0
    total = 0
    for idx, sample in enumerate(origin_data):
        if 'ID' not in sample:
            sample['ID'] = idx
        for num_hop in range(1, T + 1):
            if sample['ID'] not in rel_mapping:
                continue
            rel_mapping_dict = rel_mapping[sample['ID']]
            sample['rel_map'] = rel_mapping_dict
            if num_hop == 1:
                prev_entities = {k: {k} for k in sample['entities']}
            else:
                prev_entities = sample['entities_%d' % (num_hop - 1)]
            next_entities = dict()
            for k in prev_entities:
                if k not in rel_mapping_dict:
                    continue
                entities = prev_entities[k]
                top_relation = rel_mapping_dict[k][num_hop - 1]
                next_entities_set = set()
                for entity in entities:
                    if top_relation != 'EOD' and entity in facts and top_relation in facts[entity]:
                        next_entities_set.update(set(facts[entity][top_relation].keys()))
                next_entities[k] = next_entities_set
                if top_relation != 'EOD' and not next_entities_set:
                    err += 1
                total += 1

            # ground_truth_next_all_entities = set()
            # for gt in sample['ground_truth_path']:
            #     if num_hop * 2 < len(gt):
            #         ground_truth_next_all_entities.add(gt[num_hop * 2])
            # Choose intersection of next predicting entities from each topic entity path.
            next_entity_map_items = list(next_entities.items())
            next_entities = {k: v for k, v in next_entity_map_items}
            visited_entities = set()
            for prev_hop in range(1, num_hop):
                if ('entities_%d' % prev_hop) in sample:
                    visited_entities.update(sample['entities_%d' % prev_hop])
            next_entities = {k: {vv for vv in v if vv not in visited_entities} for k, v in next_entities.items()}
            sample['entities_%d' % num_hop] = next_entities
    # print('error_rate', err / total)
    # print('next_hop_entities mean', np.mean(item_lens))
    # print('next_hop_entities recall', avg_recall / len(origin_data))

    print('avg_loss', sum(eval_loss) / len(eval_loss))
    # print('max_acc', sum(eval_max_acc) / len(eval_max_acc))
    print('avg_hit_at_one', sum(eval_hit_at_one) / len(eval_hit_at_one))
    # print('avg_recall', sum(eval_recall) / len(eval_recall))

    return sum(eval_hit_at_one) / len(eval_hit_at_one)


def train_relreasoner(cfg, is_entity=False):
    facts = load_json(cfg['fact_data'])
    features = load_json('datasets/complexwebq/features.json')
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])
    entity2id = None
    reverse_relation2id = {v: k for k, v in relation2id.items()}
    num_hop = cfg['num_hop']

    if not is_entity:
        train_data = RelReasonerDataLoader(cfg['data_folder'] + cfg['train_data'], facts, features, num_hop,
                                           word2id, relation2id, cfg['max_query_word'], cfg['use_inverse_relation'], 1, teacher_force=True)

        valid_data = RelReasonerDataLoader(cfg['data_folder'] + cfg['dev_data'], facts, features, num_hop,
                                           word2id, relation2id, cfg['max_query_word'], cfg['use_inverse_relation'], 1, teacher_force=False)
    else:
        train_data = RelReasonerObjectDataLoader(cfg['data_folder'] + cfg['train_data'], facts, features, num_hop,
                                           word2id, relation2id, cfg['max_query_word'], cfg['use_inverse_relation'], 1, teacher_force=True)

        valid_data = RelReasonerObjectDataLoader(cfg['data_folder'] + cfg['test_data'], facts, features, num_hop,
                                           word2id, relation2id, cfg['max_query_word'], cfg['use_inverse_relation'], 1)

    my_model = get_relreasoner_model(cfg, num_hop, valid_data.num_kb_relation, 0, len(word2id), is_entity=is_entity)
    trainable_parameters = [p for p in my_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg['learning_rate'])
    for p in my_model.parameters():
        if p.requires_grad:
            print(p.name, p.numel())

    best_dev_recall = 0.0

    for epoch in range(cfg['num_epoch']):
        try:
            print('epoch', epoch)
            train_data.reset_batches(is_sequential=cfg['is_debug'])
            # Train
            my_model.train()
            my_model.teacher_force = True
            train_loss, train_hit_at_one, train_precision, train_recall, train_f1, train_max_acc = [], [], [], [], [], []
            for iteration in tqdm(range(train_data.num_data // cfg['batch_size'])):
                batch = train_data.get_batch(iteration, cfg['batch_size'], cfg['fact_dropout'])
                loss, pred, _ = my_model(batch, facts=facts, relation2id=relation2id, reverse_relation2id=reverse_relation2id)
                pred = pred.data.cpu().numpy().T.astype(np.int32)
                hit_at_one, _, recall, _, max_acc = cal_accuracy(pred, batch[-1][1:].T.astype(np.int32))
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
            eval_recall = inference_relreasoner(my_model, 20, valid_data, entity2id, relation2id, reverse_relation2id, cfg, is_entity, facts=facts)
            if not is_entity:
                if eval_recall > best_dev_recall and cfg['save_fpnet_model_file']:
                    print("saving model to", cfg['save_fpnet_model_file'])
                    torch.save(my_model.state_dict(), cfg['save_fpnet_model_file'])
                    best_dev_recall = eval_recall
            else:
                if eval_recall > best_dev_recall and cfg['save_entity_model_file']:
                    print("saving model to", cfg['save_entity_model_file'])
                    torch.save(my_model.state_dict(), cfg['save_entity_model_file'])
                    best_dev_recall = eval_recall

        except KeyboardInterrupt:
            break


def prediction_iterative_chain(cfg):
    facts = load_json(cfg['fact_data'])
    features = load_json('datasets/complexwebq/features.json')
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])
    entity2id = None
    reverse_relation2id = {v: k for k, v in relation2id.items()}
    T = cfg['num_hop']
    include_eod = cfg['eod'] if 'eod' in cfg else True
    load_model_files = [cfg['load_fpnet_model_file']]
    # load_model_files = ['model/webqsp/best_relreasoner_decoder']
    # load_model_files = ['model/webqsp/best_relreasoner_1_1',
    #                     'model/webqsp/best_relreasoner_1_2',
    #                     ]
    # load_model_files = ['model/wikimovie/best_relreasoner1_1',
    #                     'model/wikimovie/best_relreasoner1_2',
    #                     'model/wikimovie/best_relreasoner1_3',
    #                     ]

    prev_data = None
    cfg['load_fpnet_model_file'] = load_model_files[0]

    test_data = RelReasonerDataLoader(cfg['data_folder'] + cfg['test_data'], facts, features, T,
                                        word2id, relation2id, cfg['max_query_word'], cfg['use_inverse_relation'], 1, data=prev_data)

    my_model = get_relreasoner_model(cfg, T, test_data.num_kb_relation, 0, len(word2id))

    eval_recall = inference_relreasoner(my_model, cfg['test_batch_size'], test_data, entity2id, relation2id, reverse_relation2id,
                                        cfg, T=T, is_train=False, facts=facts, include_eod=include_eod)

    prev_data = test_data.origin_data

    avg_hit_at_one = 0
    avg_f1 = 0
    avg_recall = 0
    avg_precision = 0
    avg_interpretability = 0
    total_hit_at_one = 0
    total_interpretability = 0

    type_data_map = defaultdict(list)
    for e in test_data.origin_data:
        type_data_map[e['compositionality_type']].append(e)

    for ts in type_data_map:
        for e in tqdm(type_data_map[ts]):
            entities = e['entities']
            entity2answers = dict()
            final_answers = set()
            min_answers = 1000000000
            for entity in entities:
                for hop in range(T, 0, -1):
                    if 'entities_%d' % hop not in e:
                        continue
                    entity_map = e['entities_%d' % hop]
                    if entity in entity_map and entity_map[entity]:
                        entity2answers[entity] = entity_map[entity]
                        break
            for entity, answers in entity2answers.items():
                if len(answers) < min_answers:
                    min_answers = len(answers)
                    final_answers = answers
            for entity, answers in entity2answers.items():
                if len(answers & final_answers) > 0:
                    final_answers &= answers
            any_answer = list(sorted(final_answers))[0] if final_answers else None
            ground_truth_answers = set(e['answers'])
            hit_at_one = (1 if any_answer and any_answer in ground_truth_answers else 0)
            precision = 0
            for answer in final_answers:
                if answer in ground_truth_answers:
                    precision += 1
            precision = precision / len(final_answers) if len(final_answers) > 0 else 0
            recall = 0
            for gt_answer in ground_truth_answers:
                if gt_answer in final_answers:
                    recall += 1
            recall = recall / len(ground_truth_answers) if len(ground_truth_answers) > 0 else 0
            f1 = 0
            if precision + recall > 0:
                f1 = 2 * recall * precision / (precision + recall)
            avg_precision += precision
            avg_recall += recall
            avg_f1 += f1
            avg_hit_at_one += hit_at_one
            total_hit_at_one += 1

            if 'rel_chain_map' in e and e['rel_chain_map']:
                last_ground_truth_chain = e['rel_chain_map'][str(len(e['rel_chain_map']))]
                last_ground_truth_chain = {k: tuple(v['ground_truth'][0]) for k, v in last_ground_truth_chain.items()}
                if 'rel_map' in e:
                    predicted_chain = {k: tuple(v) for k, v in e['rel_map'].items()}
                    if predicted_chain == last_ground_truth_chain and hit_at_one == 1:
                        avg_interpretability += 1
                    if hit_at_one == 1:
                        total_interpretability += 1
                        # if predicted_chain != last_ground_truth_chain:
                        #     print('')

            # e['pred_answers'] = final_answers
        print(ts)
        print('avg_hit_at_one', avg_hit_at_one / total_hit_at_one)
        print('avg_interpretability',  avg_interpretability / total_interpretability)
        print('avg_precision', avg_precision / total_hit_at_one)
        print('avg_recall', avg_recall / total_hit_at_one)
        print('avg_f1', avg_f1 / total_hit_at_one)


def prediction_relreasoner(cfg):
    facts = load_fact2(cfg['fact_data'])
    facts_rel = load_fact(cfg['fact_data'])
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    reverse_relation2id = {v: k for k, v in relation2id.items()}
    num_hop = cfg['num_hop']
    max_relation = cfg['max_relation']

    # train_data = RelReasonerDataLoader(cfg['data_folder'] + cfg['train_data'], facts, num_hop,
    #                                    word2id, relation2id, cfg['max_query_word'], cfg['use_inverse_relation'], 1)
    #
    # valid_data = RelOrderReasonerDataLoader(cfg['data_folder'] + cfg['dev_data'], facts, num_hop, max_relation,
    #                                    word2id, relation2id, cfg['max_query_word'], cfg['use_inverse_relation'], 1)

    test_data = RelReasonerDataLoader(cfg['data_folder'] + cfg['test_data'], facts, num_hop,
                                      word2id, relation2id, cfg['max_query_word'], cfg['use_inverse_relation'], 1)

    my_model = get_relreasoner_model(cfg, num_hop, test_data.num_kb_relation, len(entity2id), len(word2id))

    eval_recall = inference_relreasoner(my_model, cfg['test_batch_size'], test_data, entity2id, relation2id, reverse_relation2id, cfg, is_train=False)

    test_data = RelOrderReasonerDataLoader(cfg['data_folder'] + cfg['test_data'], facts, num_hop, max_relation,
                                           word2id, relation2id, cfg['max_query_word'], cfg['use_inverse_relation'], 1,
                                           pred_data=test_data.data)
    my_model = get_relreasoner_model(cfg, num_hop, test_data.num_kb_relation, len(entity2id), len(word2id), is_order=True)

    eval_recall = inference_relreasoner(my_model, cfg['test_batch_size'], test_data, entity2id, relation2id, reverse_relation2id, cfg, is_order=True, is_train=False)

    eval_hit_at_one, eval_recall, eval_f1 = inference_answer(facts_rel, test_data.data)
    print('testing eval hit@1:', eval_hit_at_one)
    print('testing recall:', eval_recall)
    print('testing f1:', eval_f1)


def test(cfg):
    pass


def get_model(trainable_entities, facts, entity2id, relation2id, cfg, num_kb_relation, num_entities, num_vocab, num_iteration):
    word_emb_file = None if cfg['word_emb_file'] is None else cfg['data_folder'] + cfg['word_emb_file']
    entity_emb_file = None if cfg['entity_emb_file'] is None else cfg['data_folder'] + cfg['entity_emb_file']
    entity_kge_file = None if cfg['entity_kge_file'] is None else cfg['data_folder'] + cfg['entity_kge_file']
    relation_emb_file = None if cfg['relation_emb_file'] is None else cfg['data_folder'] + cfg['relation_emb_file']
    relation_kge_file = None if cfg['relation_kge_file'] is None else cfg['data_folder'] + cfg['relation_kge_file']

    my_model = use_cuda(GraftNet(trainable_entities, word_emb_file, entity_emb_file, entity_kge_file, relation_emb_file, relation_kge_file,
                                 cfg['num_layer'], num_kb_relation, num_entities, num_vocab, cfg['entity_dim'],
                                 cfg['word_dim'], cfg['kge_dim'], cfg['pagerank_lambda'], cfg['fact_scale'],
                                 cfg['lstm_dropout'], cfg['linear_dropout'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation']))

    if cfg['load_model_file'] is not None:
        print('loading model from', cfg['load_model_file'])
        pretrained_model_states = torch.load(cfg['load_model_file'])
        if word_emb_file is not None:
            del pretrained_model_states['word_embedding.weight']
        if entity_emb_file is not None:
            del pretrained_model_states['entity_embedding.weight']
        my_model.load_state_dict(pretrained_model_states, strict=False)

    return my_model


def get_relreasoner_model(cfg, num_hop, num_kb_relation, num_entities, num_vocab, is_entity=False):
    word_emb_file = None if cfg['word_emb_file'] is None else cfg['data_folder'] + cfg['word_emb_file']
    relation_emb_file = None if cfg['relation_emb_file'] is None else cfg['data_folder'] + cfg['relation_emb_file']

    if not is_entity:
        my_model = use_cuda(RelReasoner(word_emb_file, relation_emb_file,
                                        num_kb_relation, num_entities, num_vocab, num_hop, cfg['entity_dim'],
                                        cfg['word_dim'], cfg['num_layer'], cfg['lstm_dropout'], cfg['max_query_word']))
        if cfg['load_fpnet_model_file'] is not None:
            print('loading model from', cfg['load_fpnet_model_file'])
            pretrained_model_states = torch.load(cfg['load_fpnet_model_file'])
            # if word_emb_file is not None:
            #     del pretrained_model_states['word_embedding.weight']
            my_model.load_state_dict(pretrained_model_states, strict=False)
    else:
        my_model = use_cuda(EntityRelReasoner(word_emb_file, relation_emb_file,
                                        num_kb_relation, num_entities, num_vocab, num_hop, cfg['entity_dim'],
                                        cfg['word_dim'], cfg['lstm_dropout'], cfg['use_inverse_relation']))
        if 'load_entity_model_file' in cfg and cfg['load_entity_model_file'] is not None:
            print('loading model from', cfg['load_entity_model_file'])
            pretrained_model_states = torch.load(cfg['load_entity_model_file'])
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
    if '--train-answer-prediction' == sys.argv[1]:
        train_answer_prediction(CFG)
    elif '--test' == sys.argv[1]:
        test(CFG)
    elif '--train-pullfacts' == sys.argv[1]:
        train_pullfacts(CFG)
    elif '--train-relreasoner' == sys.argv[1]:
        train_relreasoner(CFG)
    elif '--train-relreasoner-entity' == sys.argv[1]:
        train_relreasoner(CFG, True)
    elif '--prediction-relreasoner' == sys.argv[1]:
        prediction_relreasoner(CFG)
    elif '--prediction-iterative-chain' == sys.argv[1]:
        prediction_iterative_chain(CFG)
    else:
        assert False, "--train or --test?"