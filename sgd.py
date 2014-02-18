
import numpy, sys
import theano
import theano.tensor as T
import cPickle
import os
import pdb

def sgd_optimizer(p,inputs,costs,train_set,updates_old=None,monitor=None,consider_constant=[],lr=0.001,num_epochs=300,save=False,output_folder=None):
  '''SGD optimizer with a similar interface to hf_optimizer.
  p: list of params wrt optimization is performed
  cost: theano scalar defining objective function
  inputs: [list] of symbolic inputs to graph. Must include targets if supervised
  updates_old: The updates dictionary for the sharedvariables. Check scan documentation for details.
  monitor: Monitoring cost. If empty, optimization cost is printed.
  consider_constant: Input to T.grad. Check RBM code for example.
  train_set: Dataset in the form of SequenceDataset
  lr: Learning rate for SGD
  '''
  best_cost = numpy.inf
  g = T.grad(costs,p,consider_constant=consider_constant)
  updates = dict((i, i - lr*j) for i, j in zip(p, g))
  if updates_old:
    updates_old.update(updates)
  else:
    updates_old = {}
    updates_old.update(updates)
  if monitor:
    f = theano.function(inputs, monitor, updates=updates_old)
  else:
    f = theano.function(inputs, costs, updates=updates_old)
  
  try:
    for u in xrange(num_epochs):
      cost = []
      for i in train_set.iterate(True): 
        cost.append(f(*i))
      print 'update %i, cost=' %(u+1), numpy.mean(cost, axis=0)
      this_cost = numpy.absolute(numpy.mean(cost, axis=0))
      if this_cost < best_cost:
        best_cost = this_cost
        print 'Best Params!'
        if save:
          best_params = [i.get_value().copy() for i in p]
          if not output_folder:
            cPickle.dump(best_params,open('best_params.pickle','w'))
          else:
            if not os.path.exists(output_folder):
              os.makedirs(output_folder)
            save_path = os.path.join(output_folder,'best_params.pickle')
            cPickle.dump(best_params,open(save_path,'w'))
      sys.stdout.flush()

  except KeyboardInterrupt: 
    print 'Training interrupted.'