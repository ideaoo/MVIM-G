import numpy as np
import pandas as pd
import sys
import math
import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import DataParallel
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import datetime
starttime = datetime.datetime.now()

epoches = 200

class TrainDataset(InMemoryDataset):
	def __init__(self, root, transform=None, pre_transform=None):
		super(TrainDataset, self).__init__(root, transform, pre_transform)
		self.data, self.slices = torch.load(self.processed_paths[0])

	@property
	def raw_file_names(self):
		return []
	
	@property
	def processed_file_names(self):
		return ['/work/eason1021/GCN_dataset/dataset_root_cna_0428/training_set.pt']
	
	def download(self):
		pass

	def process(self):
		data_list = []

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
			x = node_feature
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
			data = Data(x=x, edge_index=edge_index, y=y, num_nodes = num_nodes, train_mask=train_mask,val_mask = val_mask,  test_mask=test_mask)
			data_list.append(data)
		data, slices = self.collate(data_list)
		torch.save((data, slices), self.processed_paths[0])

training_set = TrainDataset(root='/work/eason1021/GCN_dataset/dataset_root_cna_0428/')

class GSGNet(torch.nn.Module):
	def __init__(self):
		super(GSGNet, self).__init__()
		self.conv1 = SAGEConv(training_set.num_node_features, 40)
		self.conv2 = SAGEConv(40, 24)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index

		x = self.conv1(x, edge_index)
		x = F.relu(x, inplace = True)
		x = F.dropout(x, p = 0.5,training=self.training)
		x = self.conv2(x, edge_index)
		return F.log_softmax(x, dim=1)		

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GSGNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.006, weight_decay=0.0005)
train_loader = DataLoader(training_set, batch_size = 1,shuffle=True)

def train():
	model.train()
	for data in train_loader :
		data = data.to(device)
		optimizer.zero_grad()
		F.nll_loss(model(data)[data.train_mask], data.y[data.train_mask]).backward()
		optimizer.step()

def test():
	model.eval()
	train_acc_average = 0
	val_acc_average = 0
	test_acc_average = 0
	for data in train_loader :
		data = data.to(device)
		logits, accs = model(data), []
		for _, mask in data('train_mask', 'val_mask', 'test_mask'):
			pred = logits[mask].max(1)[1]
			acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
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

torch.save(model, '/work/eason1021/GCN_dataset/build_model_cna_0428/train_model_gsg_3.pth')
endtime = datetime.datetime.now()
print (endtime - starttime)
#torch.save({'state_dict': model.state_dict()}, '/work/eason1021/GCN_dataset/build_model_wocna/train_model_gsg_3.pth.tar')


train = pd.DataFrame(columns=['train_accuracy'], data = train_acc_list)
test = pd.DataFrame(columns=['test_accuracy'], data = test_acc_list)

train.to_csv('/work/eason1021/GCN_dataset/build_model_cna_0428/train_cna_3.csv')
test.to_csv('/work/eason1021/GCN_dataset/build_model_cna_0428/test_cna_3.csv')

new_ticks = np.linspace(0, epoches, (epoches//(epoches//10))+1)
plt.xticks(new_ticks)
plt.plot(epoch_list, train_acc_list, label = 'training_accuracy')
plt.plot(epoch_list, test_acc_list, label = 'testing_accuracy')
plt.legend(loc = 'best')
plt.title('Accuracy in training progress')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
#plt.show()
plt.savefig("/work/eason1021/GCN_dataset/build_model_cna_0428/Accuracy in training progress_cna_3.png") 