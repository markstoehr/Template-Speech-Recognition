# first we load in things
# to make this basic bit work
import numpy as np

root_path = '/home/mark/Template-Speech-Recognition/'
data_path = root_path + 'Data/'
exp_path = root_path + 'Notebook/1/'
tmp_data_path = exp_path + 'data/'
import sys, os, cPickle
sys.path.append(root_path)

import template_speech_rec.get_train_data as gtrd
train_data_path = root_path+'Data/Train/'

file_indices = gtrd.get_data_files_indices(train_data_path)
syllable = np.array(['aa','r'])
avg_bgd, syllable_examples, backgrounds = gtrd.get_syllable_examples_backgrounds_files(train_data_path,
                                            file_indices,
                                                                                       syllable,
                                           
                                            num_examples=-1,
                                            verbose=True)



if __name__ == "__main__":
    main()
