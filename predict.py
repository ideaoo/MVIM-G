import numpy as np
import pandas as pd
import sys
import math
import os.path as osp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv 
from torch_geometric.nn import GCNConv
from torch_geometric.nn import DataParallel
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from ovito.io import import_file
from ovito.io import export_file
import datetime
starttime = datetime.datetime.now()
#read file
file_path = 'E:/MVIMG_size2/U/average.dump'
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

class PredictDataset(InMemoryDataset):
	def __init__(self, root, transform=None, pre_transform=None):
		super(PredictDataset, self).__init__(root, transform, pre_transform)
		self.data, self.slices = torch.load(self.processed_paths[0])

	@property
	def raw_file_names(self):
		return []
	
	@property
	def processed_file_names(self):
		return ['E:/MVIMG_size2/dataset_root/predict_set.pt']
	
	def download(self):
		pass

	def process(self):
		data_list = []

		for i in range(len(predict_content)) :
			content = predict_content[i]
			graph = predict_graph[i]
			node_feature = content[:]

			node_feature = torch.FloatTensor(node_feature).squeeze(1)
			source_nodes = graph[:,0]
			target_nodes = graph[:,1]
			edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
			x = node_feature
			num_nodes = x.shape[0]
			data = Data(x=x, edge_index=edge_index, num_nodes = num_nodes)
			data_list.append(data)
		data, slices = self.collate(data_list)
		torch.save((data, slices), self.processed_paths[0])

predict_set = PredictDataset(root='E:/MVIMG_size2/dataset_root/predict_set.pt')

class GSGNet(torch.nn.Module):
	def __init__(self):
		super(GSGNet, self).__init__()
		self.conv1 = SAGEConv(predict_set.num_node_features, 40)
		self.conv2 = SAGEConv(40, 24)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index

		x = self.conv1(x, edge_index)
		x = F.relu(x, inplace = True)
		x = F.dropout(x, p = 0.5,training=self.training)
		x = self.conv2(x, edge_index)
		#return x
		return F.log_softmax(x, dim=1)		

class GCNNet(torch.nn.Module):
	def __init__(self):
		super(GCNNet, self).__init__()
		self.conv1 = GCNConv(predict_set.num_node_features, 40)
		self.conv2 = GCNConv(40, 24)


	def forward(self, data):
		x, edge_index, batch = data.x, data.edge_index, data.batch

		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x, p = 0.5,training=self.training) 
		x = self.conv2(x, edge_index)
		#return x
		return F.log_softmax(x, dim=1)		


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
model1 = GCNNet()
model1_checkpoint = torch.load('/work/eason1021/GCN_dataset/build_model_2/train_model_gsg_1.pth.tar')
model1_checkpoint['state_dict']["module.conv1.lin.weight"] = model1_checkpoint['state_dict']["conv1.lin_l.weight"]
del model1_checkpoint['state_dict']["conv1.lin_l.weight"]
model1_checkpoint['state_dict']["module.conv2.lin.weight"] = model1_checkpoint['state_dict']["conv2.lin_l.weight"]
del model1_checkpoint['state_dict']["conv2.lin_l.weight"]
model1_checkpoint['state_dict']["module.conv1.bias"] = model1_checkpoint['state_dict']["conv1.lin_l.bias"]
del model1_checkpoint['state_dict']["conv1.lin_l.bias"]
model1_checkpoint['state_dict']["module.conv2.bias"] = model1_checkpoint['state_dict']["conv2.lin_l.bias"]
del model1_checkpoint['state_dict']["conv2.lin_l.bias"]
#from collections import OrderedDict
#new_state_dict = OrderedDict()
#for k, v in model1_checkpoint.items():
	#name = k[7:] # remove module.
#	print(k[0:-1])
	#new_state_dict[name] = v
#print(model1_checkpoint.items())
#for name in model1_checkpoint['state_dict']:
#	print(name)
#model1 = nn.DataParallel(model1)
#model1.load_state_dict(new_state_dict)
model1.to(device)
model1.eval()

model2 = GCNNet()
model2_checkpoint = torch.load('/work/eason1021/GCN_dataset/build_model_2/train_model_gsg_2.pth.tar')
model2_checkpoint['state_dict']["module.conv1.lin.weight"] = model2_checkpoint['state_dict']["conv1.lin_l.weight"]
del model2_checkpoint['state_dict']["conv1.lin_l.weight"]
model2_checkpoint['state_dict']["module.conv2.lin.weight"] = model2_checkpoint['state_dict']["conv2.lin_l.weight"]
del model2_checkpoint['state_dict']["conv2.lin_l.weight"]
model2_checkpoint['state_dict']["module.conv1.bias"] = model2_checkpoint['state_dict']["conv1.lin_l.bias"]
del model2_checkpoint['state_dict']["conv1.lin_l.bias"]
model2_checkpoint['state_dict']["module.conv2.bias"] = model2_checkpoint['state_dict']["conv2.lin_l.bias"]
del model2_checkpoint['state_dict']["conv2.lin_l.bias"]
model2 = nn.DataParallel(model2)
model2.load_state_dict(model2_checkpoint['state_dict'])
model2.to(device)
model2.eval()

model3 = GCNNet()
model3_checkpoint = torch.load('/work/eason1021/GCN_dataset/build_model_2/train_model_gsg_3.pth.tar') 
model3_checkpoint['state_dict']["module.conv1.lin.weight"] = model3_checkpoint['state_dict']["conv1.lin_l.weight"]
del model3_checkpoint['state_dict']["conv1.lin_l.weight"]
model3_checkpoint['state_dict']["module.conv2.lin.weight"] = model3_checkpoint['state_dict']["conv2.lin_l.weight"]
del model3_checkpoint['state_dict']["conv2.lin_l.weight"]
model3_checkpoint['state_dict']["module.conv1.bias"] = model3_checkpoint['state_dict']["conv1.lin_l.bias"]
del model3_checkpoint['state_dict']["conv1.lin_l.bias"]
model3_checkpoint['state_dict']["module.conv2.bias"] = model3_checkpoint['state_dict']["conv2.lin_l.bias"]
del model3_checkpoint['state_dict']["conv2.lin_l.bias"]
model3 = nn.DataParallel(model3)
model3.load_state_dict(model3_checkpoint['state_dict'])
model3.to(device)
model3.eval()

'''
#model4 = GCNNet()
#model4_checkpoint = torch.load('/work/eason1021/GCN_dataset/build_model_2/model.pth')
#model4.load_state_dict(torch.load('/work/eason1021/GCN_dataset/build_model_2/model.pth'))
#model4.to(device)
#model4.eval()
model1 = torch.load('E:/build_model_2/train_model_gsg_1.pth')
model2 = torch.load('E:/build_model_2/train_model_gsg_2.pth')
model3 = torch.load('E:/build_model_2/train_model_gsg_3.pth')

# model5 = GSGNet()
# model5_checkpoint = torch.load('/data2/GCN_dataset/build_model/train_model_gsg_2.pth.tar')
# model5.load_state_dict(model5_checkpoint['state_dict'])
# model5.to(device)
# model5.eval()

predict_loader = DataLoader(predict_set, batch_size = 1,shuffle=False)
i = 0
for data in predict_loader :
	data = data.to(device)
	pipe = node.compute(i)
	predict = voted(data, model1, model2, model3)
	result = compute_myproperty(i, data, predict)
	pipe.particles_.create_property('Variants', data = result)
	export_file(pipe, 'E:/MVIMG_size2/predict_size2_%s.dump'%i, 'lammps/dump',columns = ['Particle Identifier','Particle Type','Position.X','Position.Y','Position.Z','Variants'],frame = i)
	i = i + 1
print('Done')
endtime = datetime.datetime.now()
print (endtime - starttime)
