import numpy as np
root_path = '/home/mark/Template-Speech-Recognition/'
data_path = root_path + 'Data/Train/'
exp_path = root_path+'Experiments/080312/data_dir/'
exp_load_path = root_path+'Experiments/080312/'
import sys, os, cPickle
sys.path.append(root_path)

import template_speech_rec.test_template as tt
from npufunc import log_quantizer


timeit_import_statement = """
import numpy as np
root_path = '/home/mark/Template-Speech-Recognition/'
import sys, os, cPickle
sys.path.append(root_path)

import template_speech_rec.test_template as tt
num_features = 384
length = 20
E_test = (np.random.rand(length*num_features).reshape(num_features,length) > .6).astype(np.uint8)
bg_test = (np.random.rand(num_features)*.3 + .1).astype(np.float32)
template_test = (np.random.rand(num_features*length).reshape(num_features,length)*.9 + .05).astype(np.float32)
"""

timeit_run_statement = """
length = min(E_test.shape[1],template_test.shape[1])
sum(tt.score_template_background_section(template_test[:,:length],bg_test,E_test[:,:length]))"""

num_features = 384
length = 20
E_test = (np.random.rand(length*num_features).reshape(num_features,length) > .6).astype(np.uint8)
bg_test = (np.random.rand(num_features)*.3 + .1).astype(np.float32)
template_test = (np.random.rand(num_features*length).reshape(num_features,length)*.9 + .05).astype(np.float32)
out1 = sum(tt.score_template_background_section(template_test,bg_test,E_test))

log_template = np.log(template_test.T)
log_invtemplate = np.log(1-template_test.T)
E_test_transpose = E_test.T
out2 = tt.score_template_background_section_quantizer(log_template,
                                                      log_invtemplate,
                                                      bg_test,
                                                      E_test_transpose)



import timeit
t = timeit.Timer(timeit_run_statement,timeit_import_statement)
t.repeat(5,10000)

timeit_import_statement2 = """
import numpy as np
root_path = '/home/mark/Template-Speech-Recognition/'
import sys, os, cPickle
sys.path.append(root_path)

import template_speech_rec.test_template as tt
num_features = 384
length = 20
E_test = (np.random.rand(length*num_features).reshape(num_features,length) > .6).astype(np.uint8)
bg_test = (np.random.rand(num_features)*.3 + .1).astype(np.float32)
template_test = (np.random.rand(num_features*length).reshape(num_features,length)*.9 + .05).astype(np.float32)
log_template = np.log(template_test.T)
log_invtemplate = np.log(1-template_test.T)
E_test_transpose = E_test.T
"""

timeit_run_statement2 = """
tt.score_template_background_section_quantizer(log_template,
                                                      log_invtemplate,
                                                      bg_test,
                                                      E_test_transpose)"""


t2 = timeit.Timer(timeit_run_statement2,timeit_import_statement2)
t2.repeat(5,10000)

v1_times = [7.284235954284668,
 7.26452112197876,
 6.966034889221191,
 7.232830047607422,
 6.739763975143433]

v2_times = [1.6623570919036865,
 1.7023239135742188,
 1.7054038047790527,
 1.631303071975708,
 1.4254992008209229]


#
#

##
"""
confirmed that the quantized version basically outputs the same scores
it also is about 4 times faster

In [120]: np.mean(v1_times)/np.mean(v2_times)
Out[120]: 4.3666640898991647

In [121]: np.median(v1_times)/np.median(v2_times)
Out[121]: 4.3509484712003603

also the differences between the two outputs is negligible so I'm going to 
move over to using the new likelihood

"""

# now running the profiler
import cProfile
import pstats

cProfile.run(import cProfile
import pstats

cProfile.run(timeit_run_statement)

