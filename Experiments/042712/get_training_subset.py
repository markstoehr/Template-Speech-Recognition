import os,random
s_end_str = '_s.txt'
fls = [fl for fl in os.listdir('/home/mark/projects/Template-Speech-Recognition/Data/WavFilesTrain') if fl[-len(s_end_str):]==s_end_str]
random.shuffle(fls)

template_estimate_s_fls = fls[:500]
template_estimate_phns_fls = [fl[:-len(s_end_str)]+'_phns.txt' for fl in  fls[:500]]
template_estimate_phn_times_fls = [fl[:-len(s_end_str)]+'_phn_times.txt' for fl in  fls[:500]]

et_path_fls_path = '/home/mark/projects/Template-Speech-Recognition/Experiments/042712/'

te_s_handle = open(et_path_fls_path+'et_path_files_s.txt','w')
te_phns_handle = open(et_path_fls_path+'et_path_files_phns.txt','w')
te_phn_times_handle = open(et_path_fls_path+'et_path_files_phn_times.txt','w')

for fl_id in xrange(len(template_estimate_s_fls)):
    te_s_handle.write(template_estimate_s_fls[fl_id]+'\n')
    te_phns_handle.write(template_estimate_phns_fls[fl_id]+'\n')
    te_phn_times_handle.write(template_estimate_phn_times_fls[fl_id]+'\n')
    
    
te_s_handle.close()
te_phns_handle.close()
te_phn_times_handle.close()


########################
#
# J0 tuning parameter
#
#
########################

j0_s_fls = fls[500:1000]
j0_phns_fls = [fl[:-len(s_end_str)]+'_phns.txt' for fl in  fls[500:1000]]
j0_phn_times_fls = [fl[:-len(s_end_str)]+'_phn_times.txt' for fl in  fls[500:1000]]

j0_path_fls_path = '/home/mark/projects/Template-Speech-Recognition/Experiments/042712/'

j0_s_handle = open(j0_path_fls_path+'j0_path_files_s.txt','w')
j0_phns_handle = open(j0_path_fls_path+'j0_path_files_phns.txt','w')
j0_phn_times_handle = open(j0_path_fls_path+'j0_path_files_phn_times.txt','w')

for fl_id in xrange(len(j0_s_fls)):
    j0_s_handle.write(j0_s_fls[fl_id]+'\n')
    j0_phns_handle.write(j0_phns_fls[fl_id]+'\n')
    j0_phn_times_handle.write(j0_phn_times_fls[fl_id]+'\n')
    
    
j0_s_handle.close()
j0_phns_handle.close()
j0_phn_times_handle.close()


########################
#
# Test parts
#
#
########################

test_s_fls = fls[1000:1500]
test_phns_fls = [fl[:-len(s_end_str)]+'_phns.txt' for fl in  fls[1000:1500]]
test_phn_times_fls = [fl[:-len(s_end_str)]+'_phn_times.txt' for fl in  fls[1000:1500]]

test_path_fls_path = '/home/mark/projects/Template-Speech-Recognition/Experiments/042712/'

test_s_handle = open(test_path_fls_path+'test_path_files_s.txt','w')
test_phns_handle = open(test_path_fls_path+'test_path_files_phns.txt','w')
test_phn_times_handle = open(test_path_fls_path+'test_path_files_phn_times.txt','w')

for fl_id in xrange(len(test_s_fls)):
    test_s_handle.write(test_s_fls[fl_id]+'\n')
    test_phns_handle.write(test_phns_fls[fl_id]+'\n')
    test_phn_times_handle.write(test_phn_times_fls[fl_id]+'\n')
    
    
test_s_handle.close()
test_phns_handle.close()
test_phn_times_handle.close()

