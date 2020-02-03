import sys
import time

from relchain_predictor import RelChainPredictor
from relchain_predictor_data_loader import RelChainPredictorDataLoader
from util import *


import warnings
warnings.filterwarnings("ignore")


def train(cfg, is_order=False):
    # facts = load_fact2(cfg['fact_data'])
    T = 3
    facts = load_json(cfg['fact_data'])
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    reverse_relation2id = {v: k for k, v in relation2id.items()}
    num_hop = cfg['num_hop']

    train_data = RelChainPredictorDataLoader(cfg['data_folder'] + cfg['train_data'], facts, num_hop,
                                             word2id, relation2id, cfg['max_query_word'], cfg['use_inverse_relation'], 1)

    valid_data = RelChainPredictorDataLoader(cfg['data_folder'] + cfg['dev_data'], facts, num_hop,
                                             word2id, relation2id, cfg['max_query_word'], cfg['use_inverse_relation'], 1)

    my_model = get_relchain_predictor_model(cfg, facts, relation2id, word2id, num_hop, valid_data.num_kb_relation, len(entity2id), len(word2id), is_order=is_order)
    trainable_parameters = [p for p in my_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=1e-4)
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
            my_model.is_train = True
            train_loss, train_hit_at_one, train_precision, train_recall, train_f1, train_max_acc = [], [], [], [], [], []
            for iteration in tqdm(range(train_data.num_data // cfg['batch_size'])):
                batch = train_data.get_batch(iteration, cfg['batch_size'], cfg['fact_dropout'])
                # set stop flag to false
                for e in batch[0]:
                    e['stop_flag'] = False
                total_loss, pred, answer_dist = my_model(batch)
                # for num_hop in range(1, T + 1):
                #     my_model.num_hop = num_hop
                #     loss, pred, _ = my_model(batch)
                #     if not total_loss:
                #         total_loss = loss
                #     else:
                #         total_loss += loss
                # pred = pred.data.cpu().numpy()
                hit_at_one, _, recall, _, max_acc = cal_accuracy(pred, answer_dist)
                train_hit_at_one.append(hit_at_one)
                train_loss.append(total_loss.item())
                train_recall.append(recall)
                train_max_acc.append(max_acc)
                # back propogate
                my_model.zero_grad()
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm(my_model.parameters(), cfg['gradient_clip'])
                optimizer.step()
            print('avg_hit_at_one', sum(train_hit_at_one) / len(train_hit_at_one))
            print('avg_training_loss', sum(train_loss) / len(train_loss))
            print('max_training_acc', sum(train_max_acc) / len(train_max_acc))
            print('avg_training_recall', sum(train_recall) / len(train_recall))

            print("validating ...")
            eval_recall = inference_relchain_predictor(my_model, 20, valid_data, entity2id, reverse_relation2id, cfg, is_order)
            if eval_recall > best_dev_recall and cfg['save_fpnet_model_file']:
                print("saving model to", cfg['save_fpnet_model_file'])
                torch.save(my_model.state_dict(), cfg['save_fpnet_model_file'])
                best_dev_recall = eval_recall

        except KeyboardInterrupt:
            break


def inference_relchain_predictor(my_model, test_batch_size, data, entity2id, reverse_relation2id, cfg, facts=None, num_hop=None, is_train=True, is_order=False, log_info=False):
    # Evaluation
    my_model.eval()
    my_model.teacher_force = False
    my_model.is_train = False
    eval_hit_at_one, eval_loss, eval_recall, eval_max_acc = [], [], [], []
    id2entity = {idx: entity for entity, idx in entity2id.items()}
    data.reset_batches(is_sequential = True)
    if log_info:
        f_pred = open(cfg['pred_file'], 'w')
    for iteration in tqdm(range(data.num_data // test_batch_size)):
        batch = data.get_batch(iteration, test_batch_size, fact_dropout=0.0)
        loss, pred, answer_dist = my_model(batch)
        # pred = pred.data.cpu().numpy()
        # if not is_train and not is_order:
        #     for row in range(pred.shape[0]):
        #         cands = data.data[iteration * test_batch_size + row]['rel_cands_multi_cands']
        #         relations = []
        #         for col in range(pred.shape[1]):
        #             if pred[row][col] < len(cands):
        #                 relations.append(cands[pred[row][col]])
        #         data.data[iteration * test_batch_size + row]['pred_rel_path'] = relations
        #         top_relation = relations[0]
        #         next_hop_entities = set()
        #         key = 'entities' if num_hop == 1 else 'entities_pred%d' % (num_hop - 1)
        #         for e in data.data[iteration * test_batch_size + row][key]:
        #             if e in facts and top_relation in facts[e]:
        #                 next_hop_entities.update(list(facts[e][top_relation].keys()))
        #         next_hop_entities = list(next_hop_entities)[:8]
        #         next_hop_relations = set()
        #         for e in next_hop_entities:
        #             next_hop_relations.update(list(facts[e].keys()))
        #         next_hop_relations = list(next_hop_relations)
        #         data.data[iteration * test_batch_size + row]['entities_pred%d' % (num_hop)] = next_hop_entities
        #         data.data[iteration * test_batch_size + row]['relation_pred%d' % (num_hop)] = top_relation
        #         data.data[iteration * test_batch_size + row]['rel_cands_multi_cands'] = next_hop_relations
        # if not is_train and is_order:
        #     for row in range(pred.shape[0]):
        #         max_pred = pred[row][0]
        #         pred_path = data.local_kb_rel_path_rels[iteration * test_batch_size + row][max_pred]
        #         relations = []
        #         for rel_id in pred_path:
        #             relations.append(reverse_relation2id[rel_id])
        #         data.data[iteration * test_batch_size + row]['pred_rel_path'] = relations
        hit_at_one, _, recall, _, max_acc = cal_accuracy(pred, answer_dist)
        eval_hit_at_one.append(hit_at_one)
        eval_loss.append(loss.item())
        eval_recall.append(recall)
        eval_max_acc.append(max_acc)

    print('avg_loss', sum(eval_loss) / len(eval_loss))
    print('max_acc', sum(eval_max_acc) / len(eval_max_acc))
    print('avg_hit_at_one', sum(eval_hit_at_one) / len(eval_hit_at_one))
    print('avg_recall', sum(eval_recall) / len(eval_recall))

    return sum(eval_hit_at_one) / len(eval_hit_at_one)


def get_relchain_predictor_model(cfg, facts, relation2id, word2id, num_hop, num_kb_relation, num_entities, num_vocab, is_order=False):
    word_emb_file = None if cfg['word_emb_file'] is None else cfg['data_folder'] + cfg['word_emb_file']
    relation_emb_file = None if cfg['relation_emb_file'] is None else cfg['data_folder'] + cfg['relation_emb_file']

    my_model = use_cuda(RelChainPredictor(facts, relation2id, word2id, word_emb_file, relation_emb_file,
                                          num_kb_relation, num_entities, num_vocab, num_hop, cfg['entity_dim'],
                                          cfg['word_dim'], is_train=True))
    if cfg['load_fpnet_model_file'] is not None:
        print('loading model from', cfg['load_fpnet_model_file'])
        pretrained_model_states = torch.load(cfg['load_fpnet_model_file'])
        if word_emb_file is not None:
            del pretrained_model_states['word_embedding.weight']
        my_model.load_state_dict(pretrained_model_states, strict=False)

    return my_model


if __name__ == "__main__":
    config_file = sys.argv[2]
    CFG = get_config(config_file)
    if '--train' == sys.argv[1]:
        train(CFG)
    else:
        assert False, "--train or --test?"