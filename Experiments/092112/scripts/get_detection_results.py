import numpy as np

#root_path = '/var/tmp/stoehr/Template-Speech-Recognition/'
root_path = '/home/mark/Template-Speech-Recognition/'
data_path = root_path + 'Data/'
exp_path = root_path + 'Experiments/092112/'
tmp_data_path = exp_path+'data/'
paper_path = exp_path+'papers/'
scripts_path = exp_path+'scripts/'
import sys, os, cPickle
sys.path.append(root_path)


import template_speech_rec.test_template as tt
import template_speech_rec.bernoulli_mixture as bernoulli_mixture
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.extract_local_features as elf
import template_speech_rec.template_experiments as t_exp
import template_speech_rec.estimate_template as et

from collections import defaultdict,namedtuple


# get the test dictionary
out = open(tmp_data_path+'model_dict_list.pkl','rb')
model_dict_list = cPickle.load(out)
out.close()

# # get the model to have consistent dimensions
# for k,v in model_dict_list.items():
#     model_dict_list[k][0] = v[0].reshape(1,v[0].shape[0],
#                                          v[0].shape[1])


# out = open(tmp_data_path+'model_dict_list.pkl','wb')
# cPickle.dump(model_dict_list,out)
# out.close()



# get the numbers

file_idx = tuple(set([
            int(f[:-len('phns.npy')])
            for f in os.listdir(data_path+'Test/')
            if len(f) > len('phns.npy') and f[-len('phns.npy'):] == 'phns.npy']))

s_fnames = [data_path+'Test/'+str(i)+'s.npy' for i in file_idx]
flts_fnames = [data_path+'Train/'+str(i)+'feature_label_transitions.npy' for i in file_idx]
phns_fnames = [data_path+'Train/'+str(i)+'phns.npy' for i in file_idx]

abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02])

spread_length=3
abst_threshold=abst_threshold
fft_length=512
num_window_step_samples=80
freq_cutoff=3000
sample_rate=16000
num_window_samples=320
kernel_length=7
offset=3

syllables = np.array([['aa','r'],['p','aa'],['t','aa'],['k','aa'],['b','aa'],['d','aa'],['g','aa']])


scores_by_syllable = defaultdict(list)

out = open(tmp_data_path+'model_dict_list.pkl','rb')
alexey_model_dict_list =cPickle.load(out)
out.close()



BTemplate = namedtuple('BTemplate','log_templates log_invtemplates num_mix')

def list_of_templates_to_namedtuple(l_o_templates):
    return np.array(tuple(np.log(t) 
                 for t in l_o_templates)), np.array(tuple(
        np.log(1-t) for t in l_o_templates))

def model_dict_to_namedtuple_dict(model_dict_list):
    """
    Involves a transpose because
    time should be the 0th dimension
    features are along dimension 1
    but the original model_list_dict data structure
    had those two things switch around
    """
    named_tuple_dict = {}
    for k,v in model_dict_list.items():
        named_tuple_dict[k] = tuple(
            BTemplate(log_templates=np.array(tuple(
                        np.log(t).T.astype(np.float32)
                        for t in l_o_templates)),
                      log_invtemplates=np.array(tuple(
                        np.log(1-t).T.astype(np.float32)
                        for t in l_o_templates)),
                      num_mix=len(l_o_templates))
            for l_o_templates in v)
    return named_tuple_dict

alexey_nt_list_dict = model_dict_to_namedtuple_dict(
    alexey_model_dict_list)

i = 39
s = np.load(s_fnames[i])
phns = np.load(phns_fnames[i])
flts = np.load(flts_fnames[i])
S = esp.get_spectrogram_features(s,
                                 sample_rate,
                                 num_window_samples,
                                 num_window_step_samples,
                                 fft_length,
                                 freq_cutoff,
                                 kernel_length)
E, edge_feature_row_breaks,\
      edge_orientations = esp._edge_map_no_threshold(S)
esp._edge_map_threshold_segments(E,
                                 20,
                                 1,
                                 threshold=.7,
                                 edge_orientations = edge_orientations,
                                 edge_feature_row_breaks = edge_feature_row_breaks)

E = E.T
bg = np.clip(np.load(tmp_data_path+'avg_E_bgd.npy'),.01,.4).astype(np.float32)

k = ('t','aa')
list_of_syllable_models= alexey_nt_list_dict[k]
syllable_model = list_of_syllable_models[0]
log_template, log_invtemplate = zip(syllable_model.log_templates,syllable_model.log_invtemplates)[0]

tuple( tt.score_template_background_section_quantizer(log_template,
                                                      log_invtemplate,
                                                      bg,
                                                      E[t:t+len(log_template)],return_both_summed=True) 
       for t in xrange(len(E)-len(log_template)))

alexey_output = dict(
    # compute the score over the different syllables
    (k,tuple(
            # compute the score for each mixture number
            tuple(
                # Compute the score for each time period
                tuple( tt.score_template_background_section_quantizer(log_template,
                                                                      log_invtemplate,
                                                                  bg,
                                                                      E[t:t+len(log_template)],return_both_summed=True) 
                       for t in xrange(len(E)-len(log_template)))
                for log_template, log_invtemplate in zip(syllable_model.log_templates,syllable_model.log_invtemplates)
                )
            for syllable_model in list_of_syllable_models))
    for k,list_of_syllable_models in alexey_nt_list_dict.items())

#
# process for doing the IO to save the intermediate results 
#    
out = open(tmp_data_path+'alexey_output.pkl','wb')
cPickle.dump(alexey_output,out)
out.close()

def output_detection_scores(s_fname,phns_fname,flts_fname,
                            scores_by_syllable,model_dict_list):
    """
    Parameters:
    ===========
    model_dict_list:
        hashtable, values are lists of models, keys are syllables
        the lists of models are actually given by the number of mixture
        components
    
    Returns:
    =========
    outputs_by_syllable:
        hashtable, keys are syllables, values are also hash tables
        keys to the hash are the number of components in the mixture
        model, the values to this hash table are lists of detection
        scores over all the utterances
        
    """
    s = np.load(s_fname)
    phns = np.load(phns_fname)
    flts = np.load(flts_fname)
        S = esp.get_spectrogram_features(s,
                                     sample_rate,
                                     num_window_samples,
                                     num_window_step_samples,
                                     fft_length,
                                     freq_cutoff,
                                     kernel_length)
    E, edge_feature_row_breaks,\
      edge_orientations = esp._edge_map_no_threshold(S)
    esp._edge_map_threshold_segments(E,
                                     20,
                                     1,
                                     threshold=.7,
                                     edge_orientations = edge_orientations,
                                     edge_feature_row_breaks = edge_feature_row_breaks)
    tt.score_template_background_section_quantizer(log_template,log_invtemplate,bg,E,return_both_summed=True)
