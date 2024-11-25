from ovito.io import import_file
from ovito.io import export_file
from ovito.modifiers import *
from ovito.data import *
from ovito.data import NearestNeighborFinder
import numpy as np
import pandas as pd
import sys
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import random
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import datetime
starttime = datetime.datetime.now()

my_var = {}
def ref_neighbor(data):
	ref_list = []
	N = 6
	finder = NearestNeighborFinder(N, data)
	ptype = data.particles['Particle Type']
	
	#Loop over all input particles:
	for index in range(data.particles.count):
		neighbors = [ (neigh.index, neigh.delta) for neigh in finder.find(index) ]
		neigh_list = [0]*5
		resorted_neighbors_x_ref = sorted( neighbors , key=lambda k: [k[1][0], k[1][1], k[1][2]], reverse=True )
		resorted_neighbors_y_ref = sorted( neighbors , key=lambda k: [k[1][1], k[1][0], k[1][2]], reverse=True )
		resorted_neighbors_z_ref = sorted( neighbors , key=lambda k: [k[1][2], k[1][0], k[1][1]], reverse=True )
		resorted_neighbors_nx_ref = sorted( neighbors , key=lambda k: [k[1][0], k[1][1], k[1][2]], reverse=False )
		resorted_neighbors_ny_ref = sorted( neighbors , key=lambda k: [k[1][1], k[1][0], k[1][2]], reverse=False )
		neigh_list[0] = resorted_neighbors_x_ref[0]
		neigh_list[1] = resorted_neighbors_y_ref[0]
		neigh_list[2] = resorted_neighbors_z_ref[0]
		neigh_list[3] = resorted_neighbors_nx_ref[0]
		neigh_list[4] = resorted_neighbors_ny_ref[0]
		my_var["neigh_list%s"%index] = neigh_list

	for neigh_sort in range(data.particles.count):
		phase_index_list = [0]*7
		#+X
		phase_index_list[0] = my_var["neigh_list%s"%neigh_sort][0]
		#+Y
		phase_index_list[1] = my_var["neigh_list%s"%neigh_sort][1]
		#+Z
		phase_index_list[2] = my_var["neigh_list%s"%neigh_sort][2]
		#-X
		phase_index_list[3] = my_var["neigh_list%s"%neigh_sort][3]
		#-Y
		phase_index_list[4] = my_var["neigh_list%s"%neigh_sort][4]
		#+X+Z
		phase_index_list[5] = (my_var["neigh_list%s"%my_var['neigh_list%s'%neigh_sort][0][0]][2][0],tuple(((np.array(my_var['neigh_list%s'%my_var['neigh_list%s'%neigh_sort][0][0]][2][1]) + np.array((my_var['neigh_list%s'%neigh_sort][0][1]))).tolist())))
		#+Y+Z
		phase_index_list[6] = (my_var["neigh_list%s"%my_var['neigh_list%s'%neigh_sort][1][0]][2][0],tuple(((np.array(my_var['neigh_list%s'%my_var['neigh_list%s'%neigh_sort][1][0]][2][1]) + np.array((my_var['neigh_list%s'%neigh_sort][1][1]))).tolist())))

		ref_list.append(phase_index_list)
	return ref_list

def mvim(frame, data ,ref):
	property_list = []
	graph_list = []
	content_list = []
	N = 6
	finder = NearestNeighborFinder(N, data)
	ptype = data.particles['Particle Type']
	pid = data.particles['Particle Identifier']
	cna_result = data.particles['Structure Type'][:]
	cna_result = np.array(cna_result)
	cna_one = np.zeros((cna_result.size, 5))
	cna_one[np.arange(cna_result.size),cna_result] = 1
	#Loop over all input particles:
	for index in range(data.particles.count):
		neighbors = [ (neigh.index, neigh.delta) for neigh in finder.find(index) ]
		neigh_list = [0]*5
		resorted_neighbors_x_ref = sorted( neighbors , key=lambda k: [k[1][0], k[1][1], k[1][2]], reverse=True )
		resorted_neighbors_y_ref = sorted( neighbors , key=lambda k: [k[1][1], k[1][0], k[1][2]], reverse=True )
		resorted_neighbors_z_ref = sorted( neighbors , key=lambda k: [k[1][2], k[1][0], k[1][1]], reverse=True )
		resorted_neighbors_nx_ref = sorted( neighbors , key=lambda k: [k[1][0], k[1][1], k[1][2]], reverse=False )
		resorted_neighbors_ny_ref = sorted( neighbors , key=lambda k: [k[1][1], k[1][0], k[1][2]], reverse=False )
		neigh_list[0] = resorted_neighbors_x_ref[0]
		neigh_list[1] = resorted_neighbors_y_ref[0]
		neigh_list[2] = resorted_neighbors_z_ref[0]
		neigh_list[3] = resorted_neighbors_nx_ref[0]
		neigh_list[4] = resorted_neighbors_ny_ref[0]
		my_var["neigh_list%s"%index] = neigh_list

	for neigh_sort in range(data.particles.count):
		phase_index_list = [0]*7
		#+X
		phase_index_list[0] = my_var["neigh_list%s"%neigh_sort][0]
		#+Y
		phase_index_list[1] = my_var["neigh_list%s"%neigh_sort][1]
		#+Z
		phase_index_list[2] = my_var["neigh_list%s"%neigh_sort][2]
		#-X
		phase_index_list[3] = my_var["neigh_list%s"%neigh_sort][3]
		#-Y
		phase_index_list[4] = my_var["neigh_list%s"%neigh_sort][4]
		#+X+Z
		phase_index_list[5] = (my_var["neigh_list%s"%my_var['neigh_list%s'%neigh_sort][0][0]][2][0],tuple(((np.array(my_var['neigh_list%s'%my_var['neigh_list%s'%neigh_sort][0][0]][2][1]) + np.array((my_var['neigh_list%s'%neigh_sort][0][1]))).tolist())))
		#+Y+Z
		phase_index_list[6] = (my_var["neigh_list%s"%my_var['neigh_list%s'%neigh_sort][1][0]][2][0],tuple(((np.array(my_var['neigh_list%s'%my_var['neigh_list%s'%neigh_sort][1][0]][2][1]) + np.array((my_var['neigh_list%s'%neigh_sort][1][1]))).tolist())))

		reference = ref[neigh_sort]
		for i in range(len(phase_index_list)):
			graph_list.append(pid[neigh_sort]-1)
			graph_list.append(pid[phase_index_list[i][0]] - 1)

			content_list.append(np.array(phase_index_list[i][1][0])-np.array(reference[i][1][0]))
			content_list.append(np.array(phase_index_list[i][1][1])-np.array(reference[i][1][1]))
			content_list.append(np.array(phase_index_list[i][1][2])-np.array(reference[i][1][2]))
		for cna_class in range(cna_one.shape[1]):
			content_list.append(cna_one[neigh_sort][cna_class])
	result = (graph_list, content_list)
	return result

#read file
folder_path = '/work/eason1021/GCN_dataset/size1_test/U5'
content_list = []
graph_list = []
for dirpath, dirnames, filenames in os.walk(folder_path):

	data_path_list = []
	ref_path_list = []
	for filenames_index in filenames :
		data_path = os.path.join(dirpath, filenames_index)
		if filenames_index == 'ref.dump' :
			ref_path_list.append(data_path)
		else :
			data_path_list.append(data_path)
	print(ref_path_list)
	ref_path_list = sorted(ref_path_list, key = lambda x : (len(x), x))
	data_path_list = sorted(data_path_list, key = lambda x : (len(x), x))

	try :
		ref_path = ref_path_list[0]
		node_ref_1 = import_file(ref_path)
		node_ref_1.modifiers.append(SelectParticleTypeModifier(property='Particle Type', types={2}))
		node_ref_1.modifiers.append(DeleteSelectedParticlesModifier())
		node_ref_2 = import_file(ref_path)
		node_ref_2.modifiers.append(SelectParticleTypeModifier(property='Particle Type', types={1}))
		node_ref_2.modifiers.append(DeleteSelectedParticlesModifier())
		
		for data_index in range(len(data_path_list)) :
			data_name = data_path_list[data_index]
			print(data_name)
			node = import_file(data_name)
			node_1 = import_file(data_name)
			node_1.modifiers.append(CommonNeighborAnalysisModifier(mode = CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff))
			node_1.modifiers.append(SelectParticleTypeModifier(property='Particle Type', types={2}))
			node_1.modifiers.append(DeleteSelectedParticlesModifier())
			node_2 = import_file(data_name)
			node_2.modifiers.append(CommonNeighborAnalysisModifier(mode = CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff))
			node_2.modifiers.append(SelectParticleTypeModifier(property='Particle Type', types={1}))
			node_2.modifiers.append(DeleteSelectedParticlesModifier())
			num_1 = node_1.compute(0).particles.count
			num_2 = node_2.compute(0).particles.count
			ref_list_type1 = ref_neighbor(node_ref_1.compute())
			ref_list_type2 = ref_neighbor(node_ref_2.compute())
			result1 = mvim(0, node_1.compute(0), ref_list_type1)
			result2 = mvim(0, node_2.compute(0), ref_list_type2)
			content1 = np.array(result1[1])
			content2 = np.array(result2[1])
			content1 = content1.reshape((num_1,-1))
			content2 = content2.reshape((num_2,-1))
			content = np.row_stack((content1,content2))
			graph1 = np.array(result1[0])
			graph2 = np.array(result2[0])
			graph1 = graph1.reshape((-1,2))
			graph2 = graph2.reshape((-1,2))
			graph = np.row_stack((graph1,graph2))
			content_list.append(content)
			graph_list.append(graph)
	except IndexError :
		pass

#get the shuffle data
c = list(zip(content_list, graph_list))
content_list, graph_list = zip(*c)

#record the data length
length_list = []
for file_num in range(len(content_list)):
	if file_num == 0 :
		feature = content_list[file_num]
		length_list.append(content_list[file_num].shape[0])
	else :
		feature = np.row_stack((feature, content_list[file_num]))
		length_list.append(content_list[file_num].shape[0])

#export the feature to stand a initial standard
#feature = feature[:,:-1]
#feature_df = pd.DataFrame(feature)
init_feature = pd.read_csv('/work/eason1021/GCN_dataset/std_file_cna_0428/init.csv', index_col = 0)

#Standardization
std = StandardScaler()
init_feature = std.fit_transform(init_feature)
feature_std = std.transform(feature)
#split_data
feature_list = []
start = 0
for file_number in range(len(content_list)) :
	feature_list.append(feature_std[start:start + length_list[file_number]])
	start = start + length_list[file_number]

#split training and testing
predict_content = feature_list[:]
predict_graph = graph_list[:]

#pytorch geometric dataset
##Training dataset
class PredictDataset(InMemoryDataset):
	def __init__(self, root, transform=None, pre_transform=None):
		super(PredictDataset, self).__init__(root, transform, pre_transform)
		self.data, self.slices = torch.load(self.processed_paths[0])

	@property
	def raw_file_names(self):
		return []
	
	@property
	def processed_file_names(self):
		return ['/work/eason1021/GCN_dataset/dataset_root_cna_0428/predict_set.pt']
	
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

predict_set = PredictDataset(root='/work/eason1021/GCN_dataset/dataset_root_cna_0428/')
endtime = datetime.datetime.now()
print (endtime - starttime)