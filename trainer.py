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
from preprocessing import PreProcessor
from mlp import MLP
from sgd import SGD_Optimizer
from dataset import Dataset
import pdb

class trainer():
	def __init__(self,dataset_dir,train_list,valid_list,test_list):
		self.dataset_dir = dataset_dir
		self.lists = {}
		self.lists['train'] = train_list
		self.lists['valid'] = valid_list
		self.lists['test'] = test_list
		self.preprocessor = PreProcessor(self.dataset_dir) #Assuming everything is done here
		print 'Preparing train/valid/test splits'
		self.preprocessor.prepare_fold(self.lists['train'],self.lists['valid'],self.lists['test'])
		self.data = self.preprocessor.data
		self.targets = self.preprocessor.targets
		print 'Building model.'
		self.model = MLP(n_inputs=513,n_outputs=10,n_hidden=[50,50,50],
						 activation='sigmoid',output_layer='sigmoid',dropout_rates=[0.2,0.5,0.5])

	def train(self,):
		print 'Starting training.'
		print 'Initializing train dataset.'
		train_set = Dataset([self.data['train']],batch_size=20,targets=[self.targets['train']])
		print 'Initializing valid dataset.'
		valid_set = Dataset([self.data['valid']],batch_size=20,targets=[self.targets['valid']])
		self.optimizer = SGD_Optimizer(self.model.params,[self.model.x,self.model.y],[self.model.cost,self.model.acc],momentum=True)
		self.optimizer.train(train_set,valid_set,learning_rate=0.1,num_epochs=200,save=True,mom_rate=0.9)

if __name__=='__main__':
	test = trainer('/homes/sss31/datasets/gtzan/','/homes/sss31/datasets/gtzan/lists/train_1_of_1.txt',
		           '/homes/sss31/datasets/gtzan/lists/valid_1_of_1.txt','/homes/sss31/datasets/gtzan/lists/test_1_of_1.txt')
	test.train()