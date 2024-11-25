import numpy as np
import pandas as pd
import sys
import math
import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import dgl
from dgl import save_graphs, load_graphs
from dgl.nn.pytorch import SAGEConv
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import datetime
starttime = datetime.datetime.now()

epoches = 20

class TrainDataset(DGLDataset):
    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir='./processed/',
                 force_reload=False,
                 verbose=False):
        
        super().__init__(name='mvimg_train',
                         url=url,
                         raw_dir=raw_dir,
                         save_dir=save_dir,
                         force_reload=force_reload,
                         verbose=verbose)

    def process(self):
        
        self.graphs = []

        for i in range(len(training_content)) :
       
            content = training_content[i]
            label = training_label[i]
            graph = training_graph[i]
            node_feature = content[:]
            label = label[:,-1]
            label = np.reshape(label,(-1,1))
            node_feature = torch.FloatTensor(node_feature).squeeze(1)
            source_nodes = graph[:,0]
            target_nodes = graph[:,1]
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            
            y = torch.LongTensor(label).squeeze(1)
            
            num_nodes = len(y)
            num_train = math.floor(num_nodes * 0.85)
            num_val = math.floor(num_nodes * 0.075)
            perm = torch.randperm(num_nodes)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[perm[0:num_train]] = 1
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask[perm[num_train:num_train+num_val]] = 1
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask[perm[num_train+num_val:-1]] = 1
            
            self.graph = dgl.graph((source_nodes, target_nodes), num_nodes=len(y))
            self.graph.ndata['feat'] = node_feature
            self.graph.ndata['label'] = y

            self.graph.ndata['train_mask'] = train_mask
            self.graph.ndata['val_mask'] = val_mask
            self.graph.ndata['test_mask'] = test_mask
            
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


training_set = TrainDataset()
graph = training_set[0][0]

#training_set = TrainDataset(root='/work/eason1021/GCN_dataset/dataset_root_cna_0428/')

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
model = GSGNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.006, weight_decay=0.0005)
train_loader = GraphDataLoader(graph, batch_size = 1, shuffle=True)

def train():
	model.train()
	for data in train_loader :
		data = data.to(device)
		optimizer.zero_grad()
		F.nll_loss(model(data)[data.ndata['train_mask']], data.ndata['label'][data.ndata['train_mask']]).backward()
		optimizer.step()

def test():
	model.eval()
	train_acc_average = 0
	val_acc_average = 0
	test_acc_average = 0
	for data in train_loader :
		data = data.to(device)
		logits, accs = model(data), []
		for mask in (data.ndata['train_mask'], data.ndata['val_mask'], data.ndata['test_mask']):
			pred = logits[mask].max(1)[1]
			acc = pred.eq(data.ndata['label'][mask]).sum().item() / mask.sum().item()
			accs.append(acc)
		train_acc_average = train_acc_average + accs[0]
		val_acc_average = val_acc_average + accs[1]
		test_acc_average = test_acc_average + accs[2]
	train_acc_average = train_acc_average / len(train_loader)
	val_acc_average = val_acc_average / len(train_loader)
	test_acc_average = test_acc_average / len(train_loader)
	acc_average = [train_acc_average, val_acc_average, test_acc_average]
	return acc_average

best_val_acc = test_acc = 0
train_acc_list = []
epoch_list = []
test_acc_list = []

for epoch in range(epoches):
	train()
	train_acc, val_acc, tmp_test_acc = test()
	if val_acc > best_val_acc:
		best_val_acc = val_acc
		test_acc = tmp_test_acc
	log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
	if (epoch+1)%10 == 0 or epoch ==0:
		print(log.format(epoch + 1, train_acc, best_val_acc, test_acc))
	train_acc_list.append(train_acc)
	test_acc_list.append(tmp_test_acc)
	epoch_list.append(epoch)

torch.save(model, './build_model_dgl/train_model_gsg_1.pth')
endtime = datetime.datetime.now()
print (endtime - starttime)
#torch.save({'state_dict': model.state_dict()}, '/work/eason1021/GCN_dataset/build_model_wocna/train_model_gsg_3.pth.tar')

 
train = pd.DataFrame(columns=['train_accuracy'], data = train_acc_list)
test = pd.DataFrame(columns=['test_accuracy'], data = test_acc_list)

train.to_csv('./build_model_dgl/train_cna_1.csv')
test.to_csv('./build_model_dgl/test_cna_1.csv')

new_ticks = np.linspace(0, epoches, (epoches//(epoches//10))+1)
plt.xticks(new_ticks)
plt.plot(epoch_list, train_acc_list, label = 'training_accuracy')
plt.plot(epoch_list, test_acc_list, label = 'testing_accuracy')
plt.legend(loc = 'best')
plt.title('Accuracy in training progress')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
#plt.show()
plt.savefig("./build_model_dgl/Accuracy in training progress_cna_1.png") 