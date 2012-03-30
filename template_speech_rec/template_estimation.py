import edge_signal_proc
import numpy as np
import os
from itertools import imap


def _load_data_files(utt_name,data_dir):
    """ Returns s, phns, phn_times
    where 
    s:
        1 by n array of floats which is the signal from the TIMIT
        file
    phns:
        List of phones (strings) observed in the sequence, same
        length as phn_times
    phn_times:
        2 by L array of floats where L is the length of phns
        these are the times for the phones
    
    """
    return lambda:np.loadtxt(data_dir+'/'+utt_name+'_s.txt'),\
        np.array(open(data_dir+'/'+utt_name+'_phns.txt','r').read().split()),\
        np.loadtxt(data_dir+'/'+utt_name+'_phn_times.txt')


def get_data_files_iter():
    data_dir = '/home/mark/projects/Template-Speech-Recognition/TemplateSpeechRec/Template_Data_Files'
    os.listdir(data_dir)
    utt_names = set()
    for f in os.listdir(data_dir):
        add_name = f.split('_')[0]
        if add_name:
            utt_names.add(add_name)
    # convert the utterance names to a list
    utt_names = list(utt_names)
    return imap(lambda uname: _load_data_files(uname,
                                               data_dir),
                utt_names)

