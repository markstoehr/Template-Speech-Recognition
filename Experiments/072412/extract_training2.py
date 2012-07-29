
import numpy as np
from collections import defaultdict
import cPickle

num_data = 4619

root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'
root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'
data_path = root_path + 'Data/Train/'
import sys, os, cPickle
sys.path.append(root_path)

import template_speech_rec.template_experiments as template_exp
# first get the phn lengths and the number of examples

max_phn_lengths = defaultdict(int)
num_phn_examples = defaultdict(int)

phn_list = np.load(data_path+'phn_list.npy')
model_classes48_fold_dict = {'ux' : 'uw',
                             'axr': 'er',
                             'em': 'm' ,
                             'nx' : 'n',
                             'eng': 'ng',
                             'hv': 'hh',
                             'pcl': 'cl',
                             'tcl': 'cl',
                             'kcl': 'cl',
                             'qcl': 'cl',
                             'bcl': 'vcl',
                             'dcl': 'vcl',
                             'gcl': 'vcl',
                             'h#': 'sil',
                             'pau': 'sil',
                             '#h': 'sil'}
model_classes = {}
for phn in phn_list:
    if phn not in model_classes48_fold_dict.keys():
        model_classes[phn] = phn
    else:
        model_classes[phn] = model_classes48_fold_dict[phn]

classes_to_phns = defaultdict(list)
for phn, phn_class in model_classes.items():
    classes_to_phns[phn_class].append(phn)


    
for datum in xrange(num_data):
    phns = np.load(data_path + str(datum+1)+'phns.npy')
    ftl =  np.load(data_path + str(datum+1)+'feature_label_transitions.npy')
    for phn_id, phn in enumerate(phns):
        phn_class = model_classes[phn]
        max_phn_lengths[phn_class] = max(
            max_phn_lengths[phn_class],
            ftl[phn_id+1] - ftl[phn_id])
        num_phn_examples[phn_class] += 1

out = open(data_path+'max_phn_lengths.pkl','wb')
cPickle.dump(max_phn_lengths,out)
out.close()

out = open(data_path+'num_phn_examples.pkl','wb')
cPickle.dump(num_phn_examples,out)
out.close()

stored_bg = np.load(data_path+'averageBackground.npy')

class_array = np.array(classes_to_phns.keys())
class_to_int = dict( (v,k) for k,v in enumerate(class_array))
np.save(data_path+'class_array',class_array)
out = open(data_path + 'class_to_int.pkl','wb')
cPickle.dump(class_to_int,out)
out.close()

out = open(data_path + 'model_classes.pkl','wb')
cPickle.dump(model_classes,out)
out.close()

out = open(data_path + 'classes_to_phns.pkl','wb')
cPickle.dump(classes_to_phns,out)
out.close()

# getting the examples as training examples
# we use spread 5
for phn_class in classes_to_phns.keys():
    class_examples5 = np.empty((num_phn_examples[phn_class],
                               stored_bg.shape[0],
                               max_phn_lengths[phn_class]),
                              dtype = np.uint8)
    class_examples3 = np.empty((num_phn_examples[phn_class],
                               stored_bg.shape[0],
                               max_phn_lengths[phn_class]),
                              dtype = np.uint8)
    example_utt_id_E_loc = np.empty((num_phn_examples[phn_class],
                                     2),dtype=np.uint32)
    example_context = np.empty((num_phn_examples[phn_class],
                                2),dtype=np.uint8)
    class_examples_bg = np.empty((num_phn_examples[phn_class],
                                  stored_bg.shape[0]),
                                 dtype=np.float32)
    class_examples_lengths = np.empty(num_phn_examples[phn_class],
                                        dtype = np.uint16)
    cur_example = 0
    for datum in xrange(num_data):
        phns = np.load(data_path + str(datum+1)+'phns.npy')
        ftl =  np.load(data_path + str(datum+1)+'feature_label_transitions.npy')
        E3 = np.load(data_path + str(datum+1)+'E_spread3.npy')
        E5 = np.load(data_path + str(datum+1)+'E_spread5.npy')
        for phn_id,phn in enumerate(phns):
            datum_phn_class = model_classes[phn]
            if phn_class != datum_phn_class:
                continue
            datum_phn_loc = ftl[phn_id]
            class_examples5[cur_example] = np.hstack(
                (E5[:,datum_phn_loc:min(
                            E5.shape[1],
                            datum_phn_loc + max_phn_lengths[phn_class])],
                 np.tile(E5[:,-1],(max(datum_phn_loc+max_phn_lengths[phn_class]
                                  - E5.shape[1],0),
                                  1)).T))
            class_examples3[cur_example] = np.hstack(
                (E3[:,datum_phn_loc:min(
                            E3.shape[1],
                            datum_phn_loc + max_phn_lengths[phn_class])],
                 np.tile(E3[:,-1],(max(0,datum_phn_loc+max_phn_lengths[phn_class]
                                  - E3.shape[1]),
                                  1)).T))
            class_examples_lengths[cur_example] = ftl[phn_id+1] - datum_phn_loc
            if datum_phn_loc > 20:
                class_examples_bg[cur_example] = np.clip(np.mean(
                        E3[:,datum_phn_loc-20:
                                 datum_phn_loc],axis=1),
                                                         .1,.4)
            else:
                class_examples_bg[cur_example] = stored_bg[:]
            if phn_id >0:
                example_context[cur_example][0] = class_to_int[model_classes[phns[phn_id-1]]]
            else:
                example_context[cur_example][0] = -1
            if phn_id < phns.shape[0]:
                example_context[cur_example][1] = class_to_int[model_classes[phns[phn_id+1]]]
            else:
                example_context[cur_example][1] = -1
            example_utt_id_E_loc[cur_example][0] = datum
            example_utt_id_E_loc[cur_example][1] = datum_phn_loc
            cur_example += 1
    assert (cur_example == num_phn_examples[phn_class])
    np.save(data_path+phn_class+'class_examples3',class_examples3)
    np.save(data_path+phn_class+'class_examples5',class_examples5)
    np.save(data_path+phn_class+'class_examples_lengths',class_examples_lengths)
    np.save(data_path+phn_class+'class_examples_bgs',class_examples_bg)
    np.save(data_path+phn_class+'example_utt_id_E_loc',example_utt_id_E_loc)
    np.save(data_path+phn_class+'example_context',example_context)


