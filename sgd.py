
import numpy, sys
import theano
import theano.tensor as T
import cPickle
import os
import pdb


class SGD_Optimizer():
    def __init__(self,params,inputs,costs,updates_old=None,monitor=None,consider_constant=[]):
        """
        params: parameters of the model

        """
        self.params = params
        self.inputs = inputs
        self.costs = costs 
        self.num_costs = len(costs)
        assert (isinstance(costs,list)), "The costs given to the SGD class must be a list, even for one element."
        self.updates_old = updates_old
        self.monitor = monitor
        self.consider_constant = consider_constant
        self.build_train_fn()

    def build_train_fn(self,):
        self.lr_theano = T.scalar('lr')
        self.grad_inputs = self.inputs + [self.lr_theano]
        self.gparams = T.grad(self.costs[0],self.params,consider_constant=self.consider_constant)
        updates = dict((i, i - self.lr_theano*j) for i, j in zip(self.params, self.gparams))
        self.calc_cost = theano.function(self.inputs,self.costs)
        if self.updates_old:
            self.updates_old.update(updates)
        else:
            self.updates_old = {}
            self.updates_old.update(updates)
        if self.monitor:
            self.f = theano.function(self.grad_inputs, self.monitor, updates=self.updates_old)
        else:
            self.f = theano.function(self.grad_inputs, self.costs, updates=self.updates_old)

    def train(self,train_set,valid_set=None,learning_rate=0.1,num_epochs=500,save=False,output_folder=None,lr_update=None):
        self.best_cost = numpy.inf
        self.init_lr = learning_rate
        self.lr = numpy.array(learning_rate)
        self.output_folder = output_folder
        self.train_set = train_set
        self.valid_set = valid_set
        self.save = save
        self.lr_update = lr_update
        try:
            for u in xrange(num_epochs):
                cost = []
                for i in self.train_set.iterate(True): 
                    inputs = i + [self.lr]
                    cost.append(self.f(*inputs))
                mean_costs = numpy.mean(cost,axis=0)
                print '  Epoch %i   ' %(u+1)
                print '***Train Results***'
                for i in xrange(self.num_costs):
                    print "Cost %i: %f"%(i,mean_costs[i])

                if not valid_set:
                    this_cost = numpy.absolute(numpy.mean(cost, axis=0))
                    if this_cost < best_cost:early
                        best_cost = this_cost
                        print 'Best Params!'
                        if save:
                            self.save_model()
                    sys.stdout.flush()     
                else:
                    self.perform_validation()
                
                if lr_update:
                    self.update_lr(u+1,begin_anneal=1)

        except KeyboardInterrupt: 
            print 'Training interrupted.'
    
    def perform_validation(self,):
        cost = []
        for i in self.valid_set.iterate(True): 
            cost.append(self.calc_cost(*i))
        mean_costs = numpy.mean(cost,axis=0)
        print '***Validation Results***'
        for i in xrange(self.num_costs):
            print "Cost %i: %f"%(i,mean_costs[i])
     
        this_cost = numpy.absolute(numpy.mean(cost, axis=0))[1] #Using accuracy as metric
        if this_cost < self.best_cost:
            self.best_cost = this_cost
            print 'Best Params!'
            if self.save:
                self.save_model()

    def save_model(self,):
        best_params = [param.get_value().copy() for param in self.params]
        if not self.output_folder:
            cPickle.dump(best_params,open('best_params.pickle','w'))
        else:
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
            save_path = os.path.join(self.output_folder,'best_params.pickle')
            cPickle.dump(best_params,open(save_path,'w'))


    def update_lr(self,count,update_type='annealed',begin_anneal=500.,min_lr=0.01,decay_factor=1.2):
        if update_type=='annealed':
            scale_factor = float(begin_anneal)/count
            self.lr = self.init_lr*min(1.,scale_factor)
        if update_type=='exponential':
            new_lr = float(self.init_lr)/(decay_factor**count)
            if new_lr < min_lr:
                self.lr = min_lr
            else:
                self.lr = new_lr


