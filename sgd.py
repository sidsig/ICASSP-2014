
import numpy, sys
import theano
import theano.tensor as T
import cPickle
import os
import pdb


class SGD_Optimizer():
    def __init__(self,params,inputs,cost,train_set,valid_set=None,updates_old=None,monitor=None,consider_constant=[]):
        self.params = params
        self.inputs = inputs
        self.cost = cost
        self.train_set = train_set
        self.valid_set = valid_set
        self.updates_old = updates_old
        self.monitor = monitor
        self.consider_constant = consider_constant
        self.build_train_fn()

    def build_train_fn(self,):
        self.lr_theano = T.scalar('lr')
        self.grad_inputs = self.inputs + [self.lr_theano]
        self.gparams = T.grad(cost,self.params,consider_constant=consider_constant)
        updates = dict((i, i - self.lr_theano*j) for i, j in zip(self.params, self.gparams))
        if self.updates_old:
            self.updates_old.update(updates)
        else:
            self.updates_old = {}
            self.updates_old.update(updates)
        if self.monitor:
            self.f = theano.function(self.grad_inputs, self.monitor, updates=self.updates_old)
        else:
            self.f = theano.function(self.grad_inputs, self.cost, updates=self.updates_old)

    def train(self,train_set,valid_set=None,learning_rate=0.1,num_epochs=500,save=False,output_folder=None,lr_update=None):
        best_cost = numpy.inf
        self.init_lr = learning_rate
        self.lr = learning_rate
        try:
            for u in xrange(num_epochs):
                cost = []
                for i in train_set.iterate(True): 
                    i.append(self.lr)
                    cost.append(f(*i))
                print 'Epoch %i, cost=' %(u+1), numpy.mean(cost, axis=0)
                this_cost = numpy.absolute(numpy.mean(cost, axis=0))
                if this_cost < best_cost:
                    best_cost = this_cost
                    print 'Best Params!'
                    if save:
                        best_params = [param.get_value().copy() for param in self.params]
                        if not output_folder:
                            cPickle.dump(best_params,open('best_params.pickle','w'))
                        else:
                            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)
                            save_path = os.path.join(output_folder,'best_params.pickle')
                            cPickle.dump(best_params,open(save_path,'w'))
                sys.stdout.flush()
                self.update_lr(u+1)

        except KeyboardInterrupt: 
            print 'Training interrupted.'
    
    def update_lr(self,count,update_type='annealed',begin_anneal=500,min_lr=0.01,decay_factor=1.2):
        if update_type=='annealed':
            self.lr = self.init_lr*min(1.,begin_anneal/(float)count)
        if update_type=='exponential':
            new_lr = self.init_lr/(decay_factor**count)
            if new_lr < min_lr:
                self.lr = min_lr
            else:
                self.lr = new_lr


