# first we load in things
# to make this basic bit work
import numpy as np

root_path = '/var/tmp/stoehr/Template-Speech-Recognition/'
data_path = root_path + 'Data/'
exp_path = root_path + 'Experiments/092412/'
tmp_data_path = exp_path + 'data/'
scripts_path = exp_path + 'scripts/'
import sys, os, cPickle
sys.path.append(root_path)


##
# We assume that parts have already been computed
# we are not interested in comparing different part architectures at the moment
#

# load in the parts that we use for coding

lower_cutoff=10
num_parts = 50
# retrieve the parts
parts = np.load(tmp_data_path+'bm_templates%d_%d.npy' % (lower_cutoff,num_parts))


# perform basic transformation so its easy to use
# convert to a smaller type for our cython functions
import template_speech_rec.get_train_data as gtrd

log_part_blocks, log_invpart_blocks = gtrd.reorg_parts_for_fast_filtering(parts)
log_part_blocks = log_part_blocks.astype(np.float32)
log_invpart_blocks = log_invpart_blocks.astype(np.float32)


#
#
# now we get the examples that we are wanting
#

syllables = (('p','aa'),
             ('p','iy'),
             ('b','iy'),
             ('sh','aa'),
             ('sh','iy'),
             ('f','aa'),
             ('s','aa'),
             ('s','iy'),
             ('m','aa'),
             ('m','iy'),
             ('l','aa'),
             ('l','iy'),)


train_data_path = root_path+'Data/Train/'

file_indices = gtrd.get_data_files_indices(train_data_path)

sorted_diphones = gtrd.get_ordered_kgram_phone_list(train_data_path,file_indices,2)

avg_bgd, syllable_examples, backgrounds = gtrd.get_syllables_examples_backgrounds_files(train_data_path,
                                            file_indices,
                                            syllables,
                                            log_part_blocks,
                                            log_invpart_blocks,
                                            num_examples=-1,
                                            verbose=True)

clipped_bgd = np.clip(avg_bgd.E,.1,.4)
np.save(tmp_data_path+'clipped_bgd_101812.npy',clipped_bgd)
import template_speech_rec.estimate_template as et

for syllable,examples in syllable_examples.items():
    np.save(tmp_data_path+'%s_%s_examples.npy' % syllable,
            et.extend_examples_to_max(clipped_bgd,examples))

padded_examples_syllable_dict = dict(
    (syll,
     et.extend_examples_to_max(clipped_bgd,examples))
     for syll, examples in syllable_examples.items())

del backgrounds


# estimate mixture models
#
import template_speech_rec.bernoulli_mixture as bm

mixture_models_syllable = {}
for syllable, examples in padded_examples_syllable_dict.items():
    mixture_models_syllable[syllable] = bm.BernoulliMixture(2,examples)
    mixture_models_syllable[syllable].run_EM(.000001)
    print syllable

template_tuples_syllable = {}
for syllable, mm in mixture_models_syllable.items():
    template_tuples_syllable[syllable] = et.recover_different_length_templates(mm.affinities,padded_examples_syllable_dict[syllable],
                                                                               np.array([e.shape[0] for e in syllable_examples[syllable]]))


for syllable, template_tuples in template_tuples_syllable.items():
    for i, template in enumerate(template_tuples):
        np.save(tmp_data_path+'template_%s_%s__%d_%d.npy' % (syllable[0],
                                                             syllable[1],
                                                             len(template_tuples),
                                                             i),
                                                             template)


test_example_lengths = gtrd.get_detect_lengths(data_path+'Test/')

detection_array = np.zeros((test_example_lengths.shape[0],
                            int(test_example_lengths.max()/float(log_part_blocks.shape[1]) + .5) + 2),dtype=np.float32)

linear_filters_cs = et.construct_linear_filters(aar_mixture,
                                             clipped_bgd)
# need to state the syllable we are working with
syllable = np.array(['aa','r'])


detection_array,example_start_end_times, detection_lengths = gtrd.get_detection_scores_mixture(data_path+'Test/',                        
                                                                                         detection_array,
                                                                                         syllable,
                                                                                         linear_filters_cs,
                                                                                         log_part_blocks,
                                                                                         log_invpart_blocks,verbose=True)


