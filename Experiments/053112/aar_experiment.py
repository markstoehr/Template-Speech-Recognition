root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'

import sys, os, cPickle
sys.path.append(root_path)
import numpy as np
import template_speech_rec.template_experiments as template_exp
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.estimate_template as et
import template_speech_rec.test_template as tt
import template_speech_rec.classification as cl


edge_feature_row_breaks = np.load(root_path+'Experiments/050812/edge_feature_row_breaks.npy')
edge_orientations = np.load(root_path+'Experiments/050812/edge_orientations.npy')
abst_threshold = np.load(root_path+'Experiments/050812/abst_threshold.npy')

texp = template_exp.\
    Experiment(pattern=np.array(('aa','r')),
               data_paths_file=root_path+'Data/WavFilesTrainPaths_feverfew',
               spread_length=3,
               abst_threshold=.0001*np.ones(8),
               fft_length=512,num_window_step_samples=80,
               freq_cutoff=3000,sample_rate=16000,
               num_window_samples=320,kernel_length=7
               )

train_data_iter, tune_data_iter =\
    template_exp.get_exp_iterator(texp,train_percent=.6)

output = open('aar_train_tune_data_iter053112.pkl','wb')
cPickle.dump(tune_data_iter,output)
output.close()

aar_patterns = []
train_data_iter.reset_exp()
for datum_id in xrange(train_data_iter.num_data):
    if datum_id % 10 == 0:
        print datum_id
    if train_data_iter.next(wait_for_positive_example=True,
                            compute_patterns=True,
                            max_template_length=40):
        # the context length is 11
        for p in train_data_iter.patterns:
            pattern = p.copy()
            esp.threshold_edgemap(pattern,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
            esp.spread_edgemap(pattern,edge_feature_row_breaks,edge_orientations,spread_length=2)
            aar_patterns.append(pattern)
    else:
        break

_,_ ,\
        registered_examples,template \
        = et.simple_estimate_template(aar_patterns)

np.save('aar_template053112',template)
np.save('registered_examples_aar053112',registered_examples)
mean_background = np.load(root_path+'Experiments/050812/mean_background_liy051012.npy')

data_iter = tune_data_iter
import template_speech_rec.classification as cl
classifier = cl.Classifier(template,coarse_factor=1,bg = mean_background)

like_roc, coarse_roc = cl.get_roc_generous(data_iter, classifier,coarse_thresh=-np.inf,
                   allowed_overlap = .1,
            edge_feature_row_breaks= train_data_iter.edge_feature_row_breaks,
            edge_orientations=train_data_iter.edge_orientations)


mean_background_avg = template_exp.AverageBackground()
train_data_iter.reset_exp()
for datum_id in xrange(train_data_iter.num_data):
    if datum_id % 10 == 0:
        print datum_id
    if train_data_iter.next(wait_for_positive_example=True,
                            max_template_length=40):
        # the context length is 11
        for t in xrange(train_data_iter.E.shape[1]-30):
            E_segment = train_data_iter.E[:,t:t+30]
            mean_background_avg.add_frames(E_segment, train_data_iter.edge_feature_row_breaks,
                                           train_data_iter.edge_orientations,abst_threshold)
    else:
        break
