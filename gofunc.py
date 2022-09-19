import click as ck
import pandas as pd
from utils import Ontology
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import copy
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from itertools import cycle
import math
from aminoacids import to_onehot, MAXLEN
from dgl.nn import RelGraphConv
import dgl
from torch_utils import FastTensorDataLoader
import csv
from torch.optim.lr_scheduler import MultiStepLR


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Prediction model')
@ck.option(
    '--ont', '-ont', default='mf',
    help='Prediction model')
@ck.option(
    '--batch-size', '-bs', default=37,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=256,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--device', '-d', default='cuda:1',
    help='Device')
def main(data_root, ont, batch_size, epochs, load, device):
    go_file = f'{data_root}/go.obo'
    model_file = f'{data_root}/{ont}/go_func.th'
    terms_file = f'{data_root}/{ont}/terms.pkl'
    out_file = f'{data_root}/{ont}/predictions.pkl'

    go = Ontology(go_file, with_rels=True)

    loss_func = nn.BCELoss()
    iprs_dict, terms_dict, graph, train_nids, valid_nids, test_nids, data, labels, test_df = load_data(data_root, ont)
    n_terms = len(terms_dict)
    n_iprs = len(iprs_dict)

    subclasses = go.get_subclass_of_pairs()
    relations = go.get_relation_triples()
    subclass_pairs = []
    for go1_id, go2_id in subclasses:
        if go1_id in terms_dict and go2_id in terms_dict:
            subclass_pairs.append((terms_dict[go1_id], terms_dict[go2_id]))
    rel_dict = {}
    rel_triples = []
    for go1_id, rel, go2_id in relations:
        if go1_id in terms_dict and go2_id in terms_dict:
            if rel not in rel_dict:
                rel_dict[rel] = len(rel_dict)
            rel_triples.append((terms_dict[go1_id], rel_dict[rel], terms_dict[go2_id]))

    rel_tensor = th.LongTensor(rel_triples).to(device)
    subclass_tensor = th.LongTensor(subclass_pairs).to(device)
    
    valid_labels = labels[valid_nids].numpy()
    test_labels = labels[test_nids].numpy()

    labels = labels.to(device)

    print(valid_labels.shape)
    
    graph = graph.to(device)

    train_nids = train_nids.to(device)
    valid_nids = valid_nids.to(device)
    test_nids = test_nids.to(device)

    n_rels = len(rel_dict)
    net = GOFuncModel(n_iprs, n_terms, n_rels, device).to(device)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    train_dataloader = dgl.dataloading.DataLoader(
        graph, train_nids, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)

    valid_dataloader = dgl.dataloading.DataLoader(
        graph, valid_nids, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)

    test_dataloader = dgl.dataloading.DataLoader(
        graph, test_nids, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)
    
    
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[1, 3,], gamma=0.1)

    best_loss = 10000.0
    if not load:
        print('Training the model')
        log_file = open(f'{data_root}/train_logs.tsv', 'w')
        logger = csv.writer(log_file, delimiter='\t')
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_steps = int(math.ceil(len(train_nids) / batch_size))
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for input_nodes, output_nodes, blocks in train_dataloader:
                    bar.update(1)
                    logits = net(input_nodes, output_nodes, blocks)
                    batch_labels = labels[output_nodes]
                    loss = F.binary_cross_entropy(logits, batch_labels)
                    transc_loss = net.transc_loss(subclass_tensor, rel_tensor)
                    loss += transc_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.detach().item()
            
            train_loss /= train_steps
            
            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_nids) / batch_size))
                valid_loss = 0
                preds = []
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for input_nodes, output_nodes, blocks in valid_dataloader:
                        bar.update(1)
                        logits = net(input_nodes, output_nodes, blocks)
                        batch_labels = labels[output_nodes]
                        batch_loss = F.binary_cross_entropy(logits, batch_labels)
                        valid_loss += batch_loss.detach().item()
                        preds = np.append(preds, logits.detach().cpu().numpy())
                valid_loss /= valid_steps
                roc_auc = compute_roc(valid_labels, preds)
                print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}')
                logger.writerow([epoch, train_loss, valid_loss, roc_auc])
            if valid_loss < best_loss:
                best_loss = valid_loss
                print('Saving model')
                th.save(net.state_dict(), model_file)

            scheduler.step()
            
        log_file.close()

    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(model_file))
    net.eval()
    with th.no_grad():
        test_steps = int(math.ceil(len(test_nids) / batch_size))
        test_loss = 0
        preds = []
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for input_nodes, output_nodes, blocks in test_dataloader:
                bar.update(1)
                logits = net(input_nodes, output_nodes, blocks)
                batch_labels = labels[output_nodes]
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                test_loss += batch_loss.detach().cpu().item()
                preds = np.append(preds, logits.detach().cpu().numpy())
            test_loss /= test_steps
        preds = preds.reshape(-1, n_terms)
        roc_auc = compute_roc(test_labels, preds)
        print(f'Test Loss - {test_loss}, AUC - {roc_auc}')

    preds = list(preds)
    # Propagate scores using ontology structure
    for i in range(len(preds)):
        prop_annots = {}
        for go_id, j in terms_dict.items():
            score = preds[i][j]
            for sup_go in go.get_anchestors(go_id):
                if sup_go in prop_annots:
                    prop_annots[sup_go] = max(prop_annots[sup_go], score)
                else:
                    prop_annots[sup_go] = score
        for go_id, score in prop_annots.items():
            if go_id in terms_dict:
                preds[i][terms_dict[go_id]] = score

    test_df['preds'] = preds

    test_df.to_pickle(out_file)

    
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)
    
        
class MLPBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, layer_norm=True, dropout=0.3, activation=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class GOFuncModel(nn.Module):

    def __init__(self, nb_iprs, nb_gos, nb_rels, device, hidden_dim=1024, embed_dim=1024, margin=0.1):
        super().__init__()
        self.nb_gos = nb_gos
        self.conv = RelGraphConv(nb_iprs, 1024, 7)
        input_length = hidden_dim
        net = []
        net.append(MLPBlock(input_length, hidden_dim))
        net.append(Residual(MLPBlock(hidden_dim, hidden_dim)))
        net.append(nn.Linear(input_length, nb_gos))
        net.append(nn.Sigmoid())
        self.net = nn.Sequential(*net)

        self.embed_dim = embed_dim
        self.go_embed = nn.Embedding(nb_gos, embed_dim)
        self.go_norm = nn.BatchNorm1d(embed_dim)
        k = math.sqrt(1 / embed_dim)
        nn.init.uniform_(self.go_embed.weight, -k, k)
        self.go_rad = nn.Embedding(nb_gos, 1)
        nn.init.uniform_(self.go_rad.weight, -k, k)
        # self.go_embed.weight.requires_grad = False
        # self.go_rad.weight.requires_grad = False
        
        self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, -k, k)
        self.all_gos = th.arange(self.nb_gos).to(device)
        self.margin = margin

        
    # def forward(self, features):
    #     x = self.net(features)
    #     go_embed = self.go_embed(self.all_gos)
    #     hasFunc = self.rel_embed(self.hasFuncIndex)
    #     hasFuncGO = go_embed + hasFunc
    #     go_rad = th.abs(self.go_rad(self.all_gos).view(1, -1))
    #     x = th.matmul(x, hasFuncGO.T) + go_rad
    #     logits = th.sigmoid(x)
    #     return logits

    def forward(self, input_nodes, output_nodes, blocks):
        g = blocks[0]
        features = g.ndata['feat']['_N']
        etypes = g.edata['etypes']
        x = self.conv(g, features, etypes)
        go_embed = self.go_embed(self.all_gos)
        go_rad = th.abs(self.go_rad(self.all_gos).view(1, -1))
        logits = th.sigmoid(th.matmul(x, go_embed.T) + go_rad)
        return logits

    def transc_loss(self, subclasses, relations):
        l1 = self.subClassOfLoss(subclasses)
        l2 = self.relationsLoss(relations)
        return l1 + l2

    def subClassOfLoss(self, subclasses):
        pos_dist = self.class_dist(subclasses)
        loss = th.mean(th.relu(pos_dist - self.margin))
        return loss

    def class_dist(self, data):
        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))
        dist = th.linalg.norm(c - d, dim=1, keepdim=True) + rc - rd
        return dist

    def relationsLoss(self, data):
        c = self.go_norm(self.go_embed(data[:, 0]))
        r = self.rel_embed(data[:, 1])
        d = self.go_norm(self.go_embed(data[:, 2]))
        
        #rc = th.abs(self.go_rad(data[:, 1]))
        #rd = th.abs(self.go_rad(data[:, 2]))
        #sr = rc + rd
        # c should intersect with d + r
        dst = th.linalg.norm(c + r - d, dim=1, keepdim=True)
        loss = th.mean(th.relu(dst - self.margin))
        return loss
    
        
    
    
def load_data(data_root, ont):
    terms_df = pd.read_pickle(f'{data_root}/{ont}/terms.pkl')
    terms = terms_df['gos'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    print('Terms', len(terms))
    
    ipr_df = pd.read_pickle(f'{data_root}/{ont}/interpros.pkl')
    iprs = ipr_df['interpros'].values
    iprs_dict = {v:k for k, v in enumerate(iprs)}

    train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
    valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
    test_df = pd.read_pickle(f'{data_root}/{ont}/test_data.pkl')

    df = pd.concat([train_df, valid_df, test_df])
    graphs, nids = dgl.load_graphs(f'{data_root}/{ont}/ppi.bin')

    data, labels = get_data(df, iprs_dict, terms_dict)
    graph = graphs[0]
    graph.ndata['feat'] = data
    graph.ndata['labels'] = labels
    train_nids, valid_nids, test_nids = nids['train_nids'], nids['valid_nids'], nids['test_nids']
    return iprs_dict, terms_dict, graph, train_nids, valid_nids, test_nids, data, labels, test_df

def get_data(df, iprs_dict, terms_dict):
    data = th.zeros((len(df), len(iprs_dict)), dtype=th.float32)
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        for ipr in row.interpros:
            if ipr in iprs_dict:
                data[i, iprs_dict[ipr]] = 1
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
    return data, labels

if __name__ == '__main__':
    main()
