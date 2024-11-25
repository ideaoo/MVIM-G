import numpy as np
import pandas as pd
import sys
import math
import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
import dgl
from dgl import save_graphs, load_graphs
from dgl.nn.pytorch import SAGEConv
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F
from torch import nn
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ovito.io import import_file
from ovito.io import export_file
import datetime

starttime = datetime.datetime.now()
#read file
file_path = '/work/eason1021/GCN_dataset/size1_test/U5/averagr_sort_*.dump'
node = import_file(file_path)

def voted(data, model1, model2, model3):

	predict1 = model1(data).max(1)[1]
	predict1 = predict1.cpu().numpy()
	predict1 = np.reshape(predict1, (-1,1))
	
	predict2 = model2(data).max(1)[1]
	predict2 = predict2.cpu().numpy()
	predict2 = np.reshape(predict2, (-1,1))
	
	predict3 = model3(data).max(1)[1]
	predict3 = predict3.cpu().numpy()
	predict3 = np.reshape(predict3, (-1,1))

	predict = np.column_stack((predict1, predict2, predict3))
	result = np.zeros((predict.shape[0],1))
	for j in range(predict.shape[0]) :
		result[j] = np.argmax(np.bincount(predict[j]))
	result = np.reshape(result,(1,-1))
	result = pd.DataFrame(result)
	return result

def compute_myproperty(frame, data, result):
	variant = result.to_numpy()
	variant = variant.reshape(-1)
	variant = variant.tolist()
	return variant

class PredictDataset(DGLDataset):
    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir='/work/eason1021/GCN_dataset/dataset_root_dgl/',
                 force_reload=False,
                 verbose=False):
        
        super().__init__(name='mvimg_predict',
                         url=url,
                         raw_dir=raw_dir,
                         save_dir=save_dir,
                         force_reload=force_reload,
                         verbose=verbose)

    def process(self):
        
        self.graphs = []

        for i in range(len(predict_content)) :
       
            content = predict_content[i]
            graph = predict_graph[i]
            node_feature = content[:]
            node_feature = torch.FloatTensor(node_feature).squeeze(1)
            source_nodes = graph[:,0]
            target_nodes = graph[:,1]
        
            self.graph = dgl.graph((source_nodes, target_nodes), num_nodes=node_feature.shape[0])
            self.graph.ndata['feat'] = node_feature
            self.graphs.append(self.graph)
         
    def __getitem__(self, i):
        return self.graphs

    def __len__(self):
        return len(self.graphs)
    
    def save(self):
        # save graphs and labels
        graph_path = os.path.join(self.save_dir, self.name + '_dgl_graph.bin')
        save_graphs(graph_path, self.graphs)

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_dir, self.name + '_dgl_graph.bin')
        self.graphs = load_graphs(graph_path)
        
    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_dir, self.name + '_dgl_graph.bin')
        return os.path.exists(graph_path) 

predict_set = PredictDataset()
graph = predict_set[0][0]

class GSGNet(torch.nn.Module):
    def __init__(self):
        super(GSGNet, self).__init__()
        self.conv1 = SAGEConv(26, 40, 'mean')
        self.conv2 = SAGEConv(40, 24, 'mean')

    def forward(self, data):
        x, feat = data, data.ndata['feat']
        h = self.conv1(x, feat)
        h = F.relu(h, inplace = True)
        h = F.dropout(h, p = 0.5, training=self.training)
        h = self.conv2(x, h)
  
        return F.log_softmax(h, dim=1)	
	

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model1 = torch.load('/work/eason1021/GCN_dataset/build_model_dgl/train_model_gsg_1.pth')
model2 = torch.load('/work/eason1021/GCN_dataset/build_model_dgl/train_model_gsg_2.pth')
model3 = torch.load('/work/eason1021/GCN_dataset/build_model_dgl/train_model_gsg_3.pth')


predict_loader = GraphDataLoader(graph, batch_size = 1,shuffle=False)
i = 0

for data in predict_loader :
	data = data.to(device)
	pipe = node.compute(i)
	predict = voted(data, model1, model2, model3)
	result = compute_myproperty(i, data, predict)
	pipe.particles_.create_property('Variants', data = result)
	export_file(pipe, '/work/eason1021/GCN_dataset/size1_test/predict_dgl/predict_%s.dump'%i, 'lammps/dump',columns = ['Particle Identifier','Particle Type','Position.X','Position.Y','Position.Z','Variants'],frame = i)
	i = i + 1
print('Done')
endtime = datetime.datetime.now()
print (endtime - starttime)
