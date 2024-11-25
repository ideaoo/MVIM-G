from ovito.io import import_file
from ovito.io import export_file
from ovito.modifiers import *
from ovito.data import *
from ovito.data import NearestNeighborFinder
import numpy as np
import cupy as cp
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import math
from sklearn.preprocessing import StandardScaler
import dgl
import os
from dgl import save_graphs, load_graphs
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from torch import nn
import torch
import torch.nn as nn
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
		print(resorted_neighbors_x_ref)
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
	variant = data.particles['Variants']
	pid = data.particles['Particle Identifier']
	#cna_result = data.particles['Structure Type'][:]
	#cna_result = cp.array(cna_result)
	#cna_one = cp.zeros((cna_result.size, 5))
	#cna_one[cp.arange(cna_result.size),cna_result] = 1
	#Loop over all input particles:
	for index in range(data.particles.count):
		neighbors = [ (neigh.index, neigh.delta) for neigh in finder.find(index) ]
		#print(neighbors)
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
		phase_index_list[0] = my_var["neigh_list%s"%neigh_sort][0] #phase_index_list[0] = neigh_list[0] = resorted_neighbors_x_ref[0]
		#+Y
		phase_index_list[1] = my_var["neigh_list%s"%neigh_sort][1] #phase_index_list[1] = neigh_list[1] = resorted_neighbors_y_ref[0]
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
		
		#print(phase_index_list)
		
		reference = ref[neigh_sort]
		for i in range(len(phase_index_list)):
			graph_list.append(pid[neigh_sort]-1)
			graph_list.append(pid[phase_index_list[i][0]] - 1) 

			#content_list.append(phase_index_list[i][1][0])
			#content_list.append(phase_index_list[i][1][1])
			#content_list.append(phase_index_list[i][1][2])
			content_list.append(np.array(phase_index_list[i][1][0])-np.array(reference[i][1][0]))
			content_list.append(np.array(phase_index_list[i][1][1])-np.array(reference[i][1][1]))
			content_list.append(np.array(phase_index_list[i][1][2])-np.array(reference[i][1][2]))
		#for cna_class in range(cna_one.shape[1]):
		#	content_list.append(cna_one[neigh_sort][cna_class])
		content_list.append(variant[neigh_sort])
  
	result = (graph_list, content_list)
	return result

#read file
folder_path = 'C:/Users/smcmlab-24/Desktop/pytorch_graphsage/train1'
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
	ref_path_list = sorted(ref_path_list, key = lambda x : (len(x), x))
	data_path_list = sorted(data_path_list, key = lambda x : (len(x), x))
	print(ref_path_list)
	#ref_path_list.sort()
	#data_path_list.sort()
	try :
		ref_path = ref_path_list[0]
		print(ref_path)
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
			#node_1.modifiers.append(CommonNeighborAnalysisModifier(mode = CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff))
			node_1.modifiers.append(SelectParticleTypeModifier(property='Particle Type', types={2}))
			node_1.modifiers.append(DeleteSelectedParticlesModifier())
			node_2 = import_file(data_name)
			#node_2.modifiers.append(CommonNeighborAnalysisModifier(mode = CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff))
			node_2.modifiers.append(SelectParticleTypeModifier(property='Particle Type', types={1}))
			node_2.modifiers.append(DeleteSelectedParticlesModifier())
			num_1 = node_1.compute(0).particles.count
			num_2 = node_2.compute(0).particles.count
			ref_list_type1 = ref_neighbor(node_ref_1.compute())
			ref_list_type2 = ref_neighbor(node_ref_2.compute())
			result1 = mvim(0, node_1.compute(0), ref_list_type1)
			result2 = mvim(0, node_2.compute(0), ref_list_type2)
			content1 = np.array(result1[1])
			print(content1)
			print(type(content1))
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
feature = feature[:,:-1]
feature_df = pd.DataFrame(feature)
feature_df.to_csv('./std_file_dgl/init.csv')

#Standardization
std = StandardScaler()
feature_std = std.fit_transform(feature)
#split_data
feature_list = []
start = 0
for file_number in range(len(content_list)) :
	feature_list.append(feature_std[start:start + length_list[file_number]])
	start = start + length_list[file_number] 

#split training and testing
training_content = feature_list[:]
training_label = content_list[:]
training_graph = graph_list[:]


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


endtime = datetime.datetime.now()
print (endtime - starttime)