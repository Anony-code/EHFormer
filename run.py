import argparse
import os
import random
import sys
import time

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import MetaPath2Vec
from torch.optim import SparseAdam

from models.Transformer import GT
from utils.data import load_data, SeqDataset
from utils.pytorchtools import EarlyStopping
from utils.data import model_config, dataset_config
from torch.utils.data import DataLoader
from utils.preprocess import ego_network_sampling_with_truncate, gen_seq_hetero, feature_padding, gen_path_seq_hetero, link_extraction, gen_path_metapath2vec
from graph_coarsening.coarsen_utils import get_coarsened_graph_from_dgl
import datetime
import gc
import json
import argparse
from argparse import Namespace


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.append('utils/')
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def infer_batch(model, loader, features_list, e_feat, args):
    logits_list = []
    labels_list = []
    for batch_node, batch_label in loader:
        logits, _ = model(features_list, e_feat, batch_node, args.l2norm)
        logits_list.append(logits)
        labels_list.append(batch_label)
    logits_list = torch.cat(logits_list, dim=0)
    labels_list = torch.cat(labels_list, dim=0)
    return logits_list, labels_list


def run_model(args, hgnn_params):
    torch.use_deterministic_algorithms(True)
    seed_all(args.seed)  # fix random seed
    print(args)
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)
    checkpoint_path = 'checkpoint/' + post_fix + '/'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    device = args.device

    feats_type = args.feats_type
    features_list, adjM, labels, train_val_test_idx, dl = load_data(args.dataset, args)
    features_list = [mat2tensor(features).to(device)
                     for features in features_list]
    node_cnt = [features.shape[0] for features in features_list]
    sum_node = 0
    for x in node_cnt:
        sum_node += x
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
        hgnn_params['in_dim'] = in_dims
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros(
                    (features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3 or feats_type == 6 or feats_type == 7:
        in_dims = [features.shape[0] for features in features_list]
        hgnn_params['in_dim'] = in_dims
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(
                indices, values, torch.Size([dim, dim])).to(device)
        if feats_type == 6:
            features_list = feature_padding(features_list)
            in_dims = [features.shape[1] for features in features_list]
        if feats_type == 7:
            in_dims = [args.hidden_dim for _ in range(len(features_list))]

    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    edge2type = {}
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u, v)] = k
    for i in range(dl.nodes['total']):
        if (i, i) not in edge2type:
            edge2type[(i, i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            if (v, u) not in edge2type:
                edge2type[(v, u)] = k + 1 + len(dl.links['count'])

    g = dgl.DGLGraph(adjM + (adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    e_feat = []
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        e_feat.append(edge2type[(u, v)])
    e_feat = torch.tensor(e_feat, dtype=torch.long).to(device)

    # Graph coarsening
    if args.global_token_flag:
        coarsen_mat, g_coarsened = get_coarsened_graph_from_dgl(dl.hg, args)
    else:
        coarsen_mat, g_coarsened = (torch.tensor([]), torch.tensor([]))

    # ego-network sampling
    sub_g_list, target_list, nid_list = ego_network_sampling_with_truncate(dl.hg, k=args.ego_radius, args=args)
    sub_links = link_extraction(dl.hg, sub_g_list)
    node_seq_set = gen_seq_hetero(dl.hg, sub_g_list, target_list, nid_list, sub_links, args, seq_len=args.node_len)

    # ################## meta-path
    # # metapath = MetaPath2Vec(dl.hg, ['0-1', '1-0', '0-2', '2-0', '0-3', '3-0'], window_size=1, emb_dim=96) ##IMDB start node = 0
    # # metapath = MetaPath2Vec(dl.hg, ['1-0', '0-1', '1-2', '2-1', '1-3', '3-1'], window_size=1, emb_dim=256) ## DBLP start node = 1 meta_path_list=['0-1-0', '0-1-2', '0-1-3']
    # # metapath = MetaPath2Vec(dl.hg, ['0-1', '1-0'], window_size=1, emb_dim=32)  ## ACM
    # metapath = MetaPath2Vec(dl.hg, ['0-1', '1-0', '0-2', '2-0'], window_size=1, emb_dim=4)  ## ACM
    # mp_dataloader = DataLoader(torch.arange(dl.hg.num_nodes('0')), batch_size=64,
    #                     shuffle=True, collate_fn=metapath.sample)
    # mp_optimizer = SparseAdam(metapath.parameters(), lr=0.025)
    # for (pos_u, pos_v, neg_v) in mp_dataloader:
    #     mp_loss = metapath(pos_u, pos_v, neg_v)
    #     mp_optimizer.zero_grad()
    #     mp_loss.backward()
    #     mp_optimizer.step()
    
    # # target_m_nids = torch.LongTensor(metapath.local_to_global_nid['0'])
    # m_emb = metapath.node_embed.weight.data.cpu().numpy()
    # np.savez("meta_embedding_weights-acm-4.npz", weights=m_emb)
    # # m_emb = metapath.node_embed
    # exit()
    # ########################

    loaded_meta = np.load("meta_embedding_weights-freevase-"+str(args.metap)+".npz")
    m_emb = torch.tensor(loaded_meta['weights'])

    train_seq_list = []
    labels_list = []
    for node_seq in node_seq_set:
        # use official split in HGB
        train_seq = node_seq[train_idx]
        val_seq = node_seq[val_idx]
        test_seq = node_seq[test_idx]

        train_seq_list.append(train_seq)
        labels_list.append(labels[train_idx])

    num_classes = dl.labels_train['num_classes']
    hgnn_params['num_etypes'] = len(dl.links['count'])*2+1
    
    micro_f1 = torch.zeros(args.repeat)
    macro_f1 = torch.zeros(args.repeat)

    train_seq_ = torch.cat(train_seq_list, 0)
    labels_seq_ = torch.cat(labels_list, 0)

    train_dataset = SeqDataset(train_seq_, [], labels_seq_)
    val_dataset = SeqDataset(val_seq, [], labels[val_idx])
    test_dataset = SeqDataset(test_seq, [], torch.tensor(dl.labels_test['data'][test_idx]))

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=False,
                              num_workers=0)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=0)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=False,
                             num_workers=0)

    for i in range(args.repeat):
        net = GT(num_classes, in_dims, args.vq, args.metap, args.hidden_dim, args.num_layers,  args.num_heads, args.dropout, temper=args.temperature, meta_vec=m_emb, hg=dl.hg, coarsen_mat=coarsen_mat, args=args, hgnn_params=hgnn_params, g=g, id_dim=args.id_dim, node2type=dl.nodes['node2type']).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        # training loop
        net.train()
        print(checkpoint_path)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, delta=args.delta, save_path=(checkpoint_path + 'Transformer_{}_{}_{}.pt').format(args.dataset, args.num_layers, args.device))
        for epoch in range(args.epoch):
            t_start = time.time()
            # training
            net.train()
            train_loss_sum = 0
            for seq_batch, labels_batch in train_loader:
                seq_batch = seq_batch.to(device)
                # seq_path_batch = seq_path_batch.to(device)
                labels_batch = labels_batch.to(device)
                logits, vq_loss = net(features_list, e_feat, seq_batch, args.l2norm)
                if args.dataset == 'IMDB':
                    train_loss = F.binary_cross_entropy(torch.sigmoid(logits), labels_batch.float()) + args.vq_trade*vq_loss
                else:
                    logp = F.log_softmax(logits, 1)
                    train_loss = F.nll_loss(logp, labels_batch) + args.vq_trade*vq_loss

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                train_loss_sum += train_loss.item()

            t_end = time.time()

            # print training info
            print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, train_loss_sum, t_end-t_start))

            t_start = time.time()

            # validation
            net.eval()
            with torch.no_grad():
                logits, labels_val = infer_batch(net, val_loader, features_list, e_feat, args)
                if args.dataset == 'IMDB':
                    val_loss = F.binary_cross_entropy(torch.sigmoid(logits), labels_val.to(args.device).float())
                    pred = (logits.cpu().numpy() > 0).astype(int)
                    eval_result = dl.evaluate_valid(pred, labels_val)
                else:
                    logp = F.log_softmax(logits, 1)
                    val_loss = F.nll_loss(logp, labels_val.to(args.device))
                    pred = logits.cpu().numpy().argmax(axis=1)
                    onehot = np.eye(num_classes, dtype=np.int32)
                    pred = onehot[pred]
                    eval_result = dl.evaluate_valid(pred, F.one_hot(labels_val, num_classes=args.num_class))
                print(eval_result)
    
            scheduler.step(val_loss)
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        if args.dataset == "ACM":
            torch.save(net.state_dict(), (checkpoint_path + 'Transformer_{}_{}_{}.pt').format(args.dataset, args.num_layers, args.device))
        
        # testing with evaluate_results_nc
        print((checkpoint_path + 'Transformer_{}_{}_{}.pt').format(args.dataset, args.num_layers, args.device))
        net.load_state_dict(torch.load(
            (checkpoint_path + 'Transformer_{}_{}_{}.pt').format(args.dataset, args.num_layers, args.device)))
        print('-----------Start Test------------')
        net.eval()
        with torch.no_grad():
            logits, labels_test = infer_batch(net, test_loader, features_list, e_feat, args)
            test_logits = logits
            if args.mode == 1:
                pred = test_logits.cpu().numpy().argmax(axis=1)
                dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"{args.dataset}_{i+1}.txt")
            else:
                if args.dataset == 'IMDB':
                    pred = (logits.cpu().numpy() > 0).astype(int)
                else:
                    pred = test_logits.cpu().numpy().argmax(axis=1)
                    onehot = np.eye(num_classes, dtype=np.int32)
                    pred = onehot[pred]
                result = dl.evaluate_valid(pred, labels_test)

                print(result)
                micro_f1[i] = result['micro-f1']
                macro_f1[i] = result['macro-f1']
        print('-----------Finish Test------------')

    # empty cache
    del optimizer, net, features_list, e_feat
    gc.collect()
    torch.cuda.empty_cache()

    return micro_f1.mean().item()


def load_params(json_file):
    json_file += '.json'
    json_file = os.path.join('./configs', json_file)
    with open(json_file, 'r', encoding='utf-8') as f:
        params = json.load(f)
    return params

def dict_to_namespace(config_dict):
    return Namespace(**config_dict)

if __name__ == '__main__':

    ap = argparse.ArgumentParser(description='Transformer')
    ap.add_argument('--config', type=str, default='DBLP', help='JSON config file.')

    args = ap.parse_args()
    args = load_params(args.config)
    args = dict_to_namespace(args)

    args = dataset_config(args)
    args = model_config(args)

    args.device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')

    hgnn_params = {'num_layers': args.gnn_layers,
                   'dim': args.gnn_dim,
                   'dropout': args.gnn_dropout,
                   'num_heads': args.gnn_heads}
    hgnn_params['num_heads'] = (hgnn_params['num_layers'] - 1) * [hgnn_params['num_heads']] + [1]

    run_model(args, hgnn_params)
