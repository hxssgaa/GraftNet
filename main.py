import sys
import time

import torch
from tqdm import tqdm
import numpy as np
import torch_xla.core.xla_model as xm
import torch_xla.distributed.data_parallel as dp

from data_loader import GraftNetDataSet
from torch.utils.data import DataLoader
from graftnet import GraftNet
from util import use_cuda, save_model, load_model, get_config, load_dict, cal_accuracy, custom_collate
from util import load_documents, index_document_entities, output_pred_dist


def train(cfg):
    print("training ...")
    num_cores = 8
    devices = xm.get_xla_supported_devices(max_devices=num_cores) if num_cores != 0 else []
    print(devices)

    # prepare data
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])

    train_documents = load_documents(cfg['data_folder'] + cfg['train_documents']) if cfg['use_doc'] else None
    train_document_entity_indices, train_document_texts = index_document_entities(train_documents, word2id, entity2id, cfg['max_document_word']) if cfg['use_doc'] else None, None
    train_data = GraftNetDataSet(cfg['data_folder'] + cfg['train_data'], train_documents, train_document_entity_indices, train_document_texts, word2id, relation2id, entity2id, cfg['max_query_word'], cfg['max_document_word'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation'], cfg['fact_dropout'])
    train_loader = DataLoader(
            train_data,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=num_cores, collate_fn=custom_collate)

    if cfg['use_doc']:
        if cfg['dev_documents'] != cfg['train_documents']:
            valid_documents = load_documents(cfg['data_folder'] + cfg['dev_documents'])
            valid_document_entity_indices, valid_document_texts = index_document_entities(valid_documents, word2id, entity2id, cfg['max_document_word'])
        else:
            valid_documents = train_documents
            valid_document_entity_indices, valid_document_texts = train_document_entity_indices, train_document_texts
    else:
        valid_documents, valid_document_entity_indices, valid_document_texts = None, None, None
    valid_data = GraftNetDataSet(cfg['data_folder'] + cfg['dev_data'], valid_documents, valid_document_entity_indices, valid_document_texts, word2id, relation2id, entity2id, cfg['max_query_word'], cfg['max_document_word'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation'], cfg['fact_dropout'])
    valid_loader = DataLoader(
        valid_data,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=num_cores)

    # create model & set parameters
    my_model = get_model(cfg, train_data.num_kb_relation, len(entity2id), len(word2id))
    model_parallel = dp.DataParallel(my_model, device_ids=devices)

    trainable_parameters = [p for p in my_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg['learning_rate'])

    def train_loop_fn(model, loader, device, context):
        tracker = xm.RateTracker()

        model.train()
        for x, (index, data) in enumerate(loader):
            optimizer.zero_grad()
            loss, pred, _ = model(index, data)
            loss.backward()
            torch.nn.utils.clip_grad_norm(my_model.parameters(), cfg['gradient_clip'])
            xm.optimizer_step(optimizer)
            tracker.add(cfg['batch_size'])
            if x % 10 == 0:
                print('[{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}\n'.format(
                    device, x, loss.item(), tracker.rate(),
                    tracker.global_rate(), time.asctime()), flush=True)

    # best_dev_acc = 0.0
    # for epoch in range(cfg['num_epoch']):
    #     try:
    #         print('epoch', epoch)
    #         train_data.reset_batches(is_sequential = cfg['is_debug'])
    #         # Train
    #         my_model.train()
    #         train_loss, train_hit_at_one, train_precision, train_recall, train_f1, train_max_acc = [], [], [], [], [], []
    #         for iteration in tqdm(range(train_data.num_data // cfg['batch_size'])):
    #             batch = train_data.get_batch(iteration, cfg['batch_size'], cfg['fact_dropout'])
    #             loss, pred, _ = my_model(batch)
    #             pred = pred.data.cpu().numpy()
    #             hit_at_one, precision, recall, f1, max_acc = cal_accuracy(pred, batch[-1])
    #             train_loss.append(loss.item())
    #             train_hit_at_one.append(hit_at_one)
    #             train_precision.append(precision)
    #             train_recall.append(recall)
    #             train_f1.append(f1)
    #             train_max_acc.append(max_acc)
    #             # back propogate
    #             my_model.zero_grad()
    #             optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm(my_model.parameters(), cfg['gradient_clip'])
    #             xm.optimizer_step(optimizer)
    #         print('avg_training_loss', sum(train_loss) / len(train_loss))
    #         print('max_training_acc', sum(train_max_acc) / len(train_max_acc))
    #         print('avg_training_hit@1', sum(train_hit_at_one) / len(train_hit_at_one))
    #         print('avg_training_precision', sum(train_precision) / len(train_precision))
    #         print('avg_training_recall', sum(train_recall) / len(train_recall))
    #         print('avg_training_f1', sum(train_f1) / len(train_f1))
    #
    #         print("validating ...")
    #         eval_acc = inference(my_model, valid_data, entity2id, cfg)
    #         if eval_acc > best_dev_acc and cfg['to_save_model']:
    #             print("saving model to", cfg['save_model_file'])
    #             torch.save(my_model.state_dict(), cfg['save_model_file'])
    #             best_dev_acc = eval_acc
    #
    #     except KeyboardInterrupt:
    #         break


    # for x, data in enumerate(train_loader):
    #     optimizer.zero_grad()
    #     loss, pred, _ = my_model(data)
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm(my_model.parameters(), cfg['gradient_clip'])
    #     # xm.optimizer_step(optimizer)
    #     optimizer.step()
    #     if x % 10 == 0:
    #         print(loss.item())

    for epoch in range(1, cfg['num_epoch'] + 1):
        model_parallel(train_loop_fn, train_loader)
    # Test set evaluation
    # print("evaluating on test")
    # print('loading model from ...', cfg['save_model_file'])
    # my_model.load_state_dict(torch.load(cfg['save_model_file']))
    # test_acc = inference(my_model, test_data, entity2id, cfg, log_info=True)

    return None


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

    return sum(eval_precision) / len(eval_precision)

def test(cfg):
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])

    test_documents = load_documents(cfg['data_folder'] + cfg['test_documents']) if cfg['use_doc'] else None
    test_document_entity_indices, test_document_texts = index_document_entities(test_documents, word2id, entity2id, cfg['max_document_word']) if cfg['use_doc'] else None, None
    test_data = DataLoader(cfg['data_folder'] + cfg['test_data'], test_documents, test_document_entity_indices, test_document_texts, word2id, relation2id, entity2id, cfg['max_query_word'], cfg['max_document_word'], cfg['use_kb'], cfg['use_doc'], cfg['use_inverse_relation'])

    my_model = get_model(cfg, test_data.num_kb_relation, len(entity2id), len(word2id))
    test_acc = inference(my_model, test_data, entity2id, cfg, log_info=True)
    return test_acc


def get_model(cfg, num_kb_relation, num_entities, num_vocab):
    word_emb_file = None if cfg['word_emb_file'] is None else cfg['data_folder'] + cfg['word_emb_file']
    entity_emb_file = None if cfg['entity_emb_file'] is None else cfg['data_folder'] + cfg['entity_emb_file']
    entity_kge_file = None if cfg['entity_kge_file'] is None else cfg['data_folder'] + cfg['entity_kge_file']
    relation_emb_file = None if cfg['relation_emb_file'] is None else cfg['data_folder'] + cfg['relation_emb_file']
    relation_kge_file = None if cfg['relation_kge_file'] is None else cfg['data_folder'] + cfg['relation_kge_file']
    
    my_model = use_cuda(GraftNet(word_emb_file, entity_emb_file, entity_kge_file, relation_emb_file, relation_kge_file, cfg['num_layer'], num_kb_relation, num_entities, num_vocab, cfg['entity_dim'], cfg['word_dim'], cfg['kge_dim'], cfg['pagerank_lambda'], cfg['fact_scale'], cfg['lstm_dropout'], cfg['linear_dropout'], cfg['use_kb'], cfg['use_doc'])) 

    if cfg['load_model_file'] is not None:
        print('loading model from', cfg['load_model_file'])
        pretrained_model_states = torch.load(cfg['load_model_file'])
        if word_emb_file is not None:
            del pretrained_model_states['word_embedding.weight']
        if entity_emb_file is not None:
            del pretrained_model_states['entity_embedding.weight']
        my_model.load_state_dict(pretrained_model_states, strict=False)
    
    return my_model

if __name__ == "__main__":
    config_file = sys.argv[2]
    CFG = get_config(config_file)
    if '--train' == sys.argv[1]:
        train(CFG)
    elif '--test' == sys.argv[1]:
        test(CFG)
    else:
        assert False, "--train or --test?"

