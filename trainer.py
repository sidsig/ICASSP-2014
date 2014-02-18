"""
Trainer script
Siddharth Sigia
Feb,2014
C4DM
"""
import os
import sys
import numpy
import theano
import pickle
from preprocessing import PreProcesser
from mlp import MLP
from sgd import sgd_optimizer

class trainer():
	def __init__(self,dataset_dir,train_list,valid_list,test_list):
		self.dataset_dir = dataset_dir
		self.lists = {}
		self.lists['train'] = train_list
		self.lists['valid'] = valid_list
		self.lists['test'] = test_list
		self.preprocessor = PreProcesser(self.dataset_dir) #Assuming everything is done here
		self.data = self.preprocessor.data
		self.targets = self.preprocessor.targets
		self.model = MLP(n_inputs=513,n_outputs=10,n_hidden=[50,50,50],
						 activation='sigmoid',output_layer='sigmoid')

	def train(self,):
		inputs,cost,params = self.model.build_graph()
		sgd_optimizer(params,inputs,cost,lr=0.1)
