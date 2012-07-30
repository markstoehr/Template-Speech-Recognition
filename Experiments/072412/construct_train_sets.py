import numpy as np
from collections import defaultdict
root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'
data_path = root_path + 'Data/Train/'
exp_path = root_path+'Experiments/072412/'
import sys, os, cPickle
sys.path.append(root_path)

out = open(data_path+'num_phn_examples.pkl','rb')
num_phn_examples = cPickle.load(out)
out.close()

class_array = np.load(data_path+'class_array.npy')

train_percent = .7
num_folds = 10
for phn_class in class_array:
    print phn_class
    num_examples = num_phn_examples[phn_class]
    train_masks = np.zeros((num_folds,
                            num_examples),
                           dtype=bool)
    for fold_id in xrange(num_folds):
        train_masks[fold_id][
            np.random.permutation(num_examples)[:int(.6*num_examples)]] = True
    np.save(exp_path+phn_class+'_train_masks',train_masks)
                            

for phn_class in class_array:
    train_masks = np.load(exp_path+phn_class+'_train_masks.npy')
    train_phn_examples =  np.load(data_path+phn_class+"class_examples5.npy")
    train_phn_examples
