#!/usr/bin/python
import numpy as np
import os, itertools, argparse
from collections import defaultdict
import template_speech_rec.main_multiprocessing as mm


def get_fls_with_ext(files,ext):
    return sorted([f for f in files if f[-len(ext):]==ext])

def main(args):
    print args
    leehon_mapping,use_phns=mm.get_leehon_mapping()
    phone_dict = defaultdict(int)
    for root,dirs,files in os.walk(args.use_directory):
        sorted_phn_fls = get_fls_with_ext(files,'phn')
        for phn_fl in sorted_phn_fls:
            phn_triples = tuple(f.split() for f in open('%s/%s' % (root,phn_fl),'r').read().split('\n') if len(f) > 0)
            for phn_triple in phn_triples:
                phone_dict[leehon_mapping[phn_triple[2]]] += 1

    all_phn_str=''
    phns =[]
    for phn, phn_count in sorted(phone_dict.items(),key=lambda x:x[1]):
        print "%s: %d" %(phn,phn_count)
        all_phn_str+=' %s' % phn
        phns.append(phn)

    for phn in phns:
        print "\n"
        print "../../template_speech_rec/main_multiprocessing.py --detect_object %s --template_tag %s_edge --savedir data/ --save_tag %s_edge --num_mix_parallel 1 2 3 4 5 6 7 8 9 --root_path /var/tmp/stoehr/Temlate-Speech-Recognition/ --save_detection_setup test --old_max_detect_tag %s_edge" % (phn,phn,phn,phn)
#        print "../../template_speech_rec/main_multiprocessing.py --detect_object %s --save_syllable_features_to_data_dir --savedir data/ --save_tag %s_edge" % (phn,phn)
#        print "../../template_speech_rec/main_multiprocessing.py --detect_object %s --estimate_templates --num_mix_parallel 1 2 3 4 5 6 7 8 9 --savedir data/ --save_tag %s_edge --template_tag %s_edge" % (phn,phn,phn)
#        print "rm data/*lengths*"
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser("A program to count the phones in timit")
    parser.add_argument('-d','--use_directory',
                        type=str,metavar="PATH",
                        help="path to the directory we are going to use",
                        default='../../Data/TIMIT/test/')
    parser.add_argument('--use_leehon',action='store_true',
                        help='include flag if the classification set should be mapped according to the Lee and Hon standard mapping')
    main(parser.parse_args())
