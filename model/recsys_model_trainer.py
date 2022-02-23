import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score




df_node = pd.read_csv("/Users/choeseung-won/Deep_Study/Business_Partner_Project/Data/강원_node.csv")
df_edge = pd.read_csv("/Users/choeseung-won/Deep_Study/Business_Partner_Project/Data/강원_edge.csv")

df_node.Id = df_node.Id.fillna(0).astype(int)
df_edge = df_edge[['Source', 'Target']].fillna(0).astype(int)

# 전처리
df_node = df_node[df_node['Id'].isin(df_edge.Source) | df_node['Id'].isin(df_edge.Target)]
df_node = df_node.drop_duplicates('Id').rename(columns={ "from_업종명10차": "sector"}).iloc[:, :3]
# df_node['company_id'] = df_node.reindex().index
df_node = df_node.drop_duplicates('Id').reset_index()

# df_edge = df_edge[['Source', 'Target']].fillna(0).astype(int)
df_edge['Source'] = df_edge.Source.map(lambda x: df_node[df_node['Id'] == x].index[0])
df_edge['Target'] = df_edge.Target.map(lambda x: df_node[df_node['Id'] == x].index[0])
# 자기자신
df_edge = df_edge.append(pd.DataFrame({'Source': df_node.index.to_numpy(), 'Target': df_node.index.to_numpy()}))

# One - Hot
df_node = pd.get_dummies(df_node,prefix='', prefix_sep='', columns=['sector'])


## Build Graph
u, v = torch.LongTensor(df_edge.Source.to_numpy()), torch.LongTensor(df_edge.Target.to_numpy())
graph = dgl.graph((u, v))
# g.ndata['sector'] = torch.ones(g.num_nodes(), 16)
graph.ndata['sector'] = torch.FloatTensor(df_node.iloc[:,3:].to_numpy())
graph



## Define Dataloader
train, val, test = dgl.data.utils.split_dataset(graph, frac_list=[0.8, 0.1, 0.1])
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
neg_sampler = dgl.dataloading.negative_sampler.Uniform(2)

E = graph.num_nodes()
reverse_eids = torch.cat([torch.arange(E, 2 * E), torch.arange(0, E)])

test_ids = torch.arange(graph.num_edges())[:300]
train_ids = torch.arange(graph.num_edges())[300:]

train_collator = dgl.dataloading.EdgeCollator(
    graph, 
    train_ids, 
    sampler, 
    # exclude='reverse_id',
    # reverse_eids=reverse_eids, 
    negative_sampler=neg_sampler,)

train_dataloader = torch.utils.data.DataLoader(
    train_collator.dataset, 
    collate_fn=train_collator.collate,
    batch_size=1024, 
    shuffle=True, 
    drop_last=False, 
    num_workers=0)

test_collator = dgl.dataloading.EdgeCollator(
    graph, 
    test_ids, 
    sampler, 
    # exclude='reverse_id',
    # reverse_eids=reverse_eids, 
    negative_sampler=neg_sampler,)

test_dataloader = torch.utils.data.DataLoader(
    test_collator.dataset, 
    collate_fn=test_collator.collate,
    batch_size=1024, 
    shuffle=True, 
    drop_last=False, 



class StochasticTwoLayerGraphSage(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = dgl.nn.SAGEConv(in_features, hidden_features, 'mean')
        self.conv2 = dgl.nn.SAGEConv(hidden_features, out_features, 'mean')

    def forward(self, blocks, x):
        x = F.relu(self.conv1(blocks[0], x))
        x = F.relu(self.conv2(blocks[1], x))
        return x


class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            edge_subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'))
            return edge_subgraph.edata['score']


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = StochasticTwoLayerGraphSage(
            in_features, hidden_features, out_features)
        
        self.predictor = ScorePredictor()

    def forward(self, positive_graph, negative_graph, blocks, x):
        x = self.sage(blocks, x)
        pos_score = self.predictor(positive_graph, x)
        neg_score = self.predictor(negative_graph, x)
        return pos_score, neg_score



def compute_loss(pos_score, neg_score):
    # an example hinge loss
    n = pos_score.shape[0]
    return (neg_score.view(n, -1) - pos_score.view(n, -1) + 1).clamp(min=0).mean()

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


node_features = graph.ndata['sector']
model = Model(node_features.shape[1], 128, 8)
opt = torch.optim.Adam(model.parameters())


for e in range(200):
    for input_nodes, positive_graph, negative_graph, blocks in train_dataloader:
        blocks = [b for b in blocks]
        positive_graph = positive_graph
        negative_graph = negative_graph
        input_features = blocks[0].srcdata['sector']

        pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)
        loss = compute_loss(pos_score, neg_score)

        opt.zero_grad()
        loss.backward()
        opt.step()
    
    if e % 10 == 0:
        print('In epoch {}, train_loss: {}'.format(e, loss))
        with torch.no_grad():
            for input_nodes, positive_graph, negative_graph, blocks in test_dataloader:
                blocks = [b for b in blocks]
                positive_graph = positive_graph
                negative_graph = negative_graph
                input_features = blocks[0].srcdata['sector']

                pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)
                loss = compute_loss(pos_score, neg_score)
                print('In epoch {}, val_loss: {}'.format(e, loss))
                print('='*100)
# Test
with torch.no_grad():
    for input_nodes, positive_graph, negative_graph, blocks in test_dataloader:
        blocks = [b for b in blocks]
        positive_graph = positive_graph
        negative_graph = negative_graph
        input_features = blocks[0].srcdata['sector']
        pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)

        print('AUC', compute_auc(pos_score, neg_score))
        
        
torch.save(model.state_dict(), f'./model_test_epoch{epoch}.pt')
