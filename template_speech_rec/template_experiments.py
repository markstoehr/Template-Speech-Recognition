import numpy as np
import edge_signal_proc as esp
import test_template as tt
import random




class Experiment:
    def __init__(self,patterns,data_paths_file,bg_len=26,
                 offset=3,
                 sample_rate=16000,freq_cutoff=3000,
                 num_window_samples=320,
                 num_window_step_samples=80,
                 fft_length=512,
                 data_dir='',kernel_length=7,
                 spread_length=3,
                 abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02]),
                 do_random=True,
                 edge_feature_row_breaks=np.array([   0.,   
                                                  45.,   
                                               90.,  
                                               138.,  
                                               186.,  
                                               231.,  
                                               276.,  
                                               321.,  
                                               366.]),
                 edge_orientations=np.array([[ 1.,  0.],
                                    [-1.,  0.],
                                        [ 0.,  1.],
                                        [ 0., -1.],
                                        [ 1.,  1.],
                                        [-1., -1.],
                                        [ 1., -1.],
                                [-1.,  1.]]),
                 use_mel=False
                 ):
        """
        Parameters:
        -----------
        data_paths_file: string
            The format of this file should be just the front of 
            the paths without _s.txt, _phns.txt, _phn_times.txt
            which need to be added in the algorithm
        sample_rate:
            
        freq_cutoff:
            
        num_window_samples:
            
        num_window_step_samples:
        fft_length:
        data_dir:

        
        """
        self.offset = offset
        self.bg_len = bg_len
        self.spread_length = spread_length
        self.abst_threshold = abst_threshold
        self.kernel_length = kernel_length
        self.patterns = patterns
        self.data_dir = data_dir
        # make sure that the data dir ends with /
        self.sample_rate=sample_rate
        self.freq_cutoff = freq_cutoff
        self.num_window_samples = num_window_samples
        self.num_window_step_samples = num_window_step_samples
        self.fft_length = fft_length
        # assume that the files were generated intelligently so there is a correspondence between the labels
        self.data_paths_file = data_paths_file
        self.paths =  open(data_paths_file,'r').read().strip('\n').split('\n')
        self.num_data = len(self.paths)    
        if do_random:
            random.shuffle(self.paths)
        self.edge_feature_row_breaks= edge_feature_row_breaks
        self.edge_orientations=edge_orientations
        self.abst_threshold=abst_threshold
        self.spread_radius=spread_length
        self.use_mel = use_mel



    def get_s(self,idx):
        return np.loadtxt(self.paths[idx] + '_s.txt')

    def get_phns(self,idx):
        return np.array([phn for phn in \
                    open(self.paths[idx]+'_phns.txt','r').read().split('\n') \
                    if phn != ''])


    def get_phn_times(self,idx):
        return np.loadtxt(self.paths[idx]+'_phn_times.txt')


    def frame_to_phn_idx(self,frame_num,s,phn_times,phns):
        feature_start, \
            feature_step, num_features =\
            esp._get_feature_label_times(s,
                                     self.num_window_samples,
                                     self.num_window_step_samples)
        feature_labels, \
            feature_label_transitions \
            = esp._get_labels(phn_times,
                          phns,
                          feature_start,
                          feature_step,
                          num_features,
                          self.sample_rate)
        feature_label_transitions < frame_num
        
    def get_edgemap_no_threshold(self,s):
        return esp.get_edgemap_no_threshold(s,self.sample_rate,
                                             self.num_window_samples,
                                             self.num_window_step_samples,
                                             self.fft_length,
                                             self.freq_cutoff,
                                             self.kernel_length,
                                            use_mel=self.use_mel)
        
    def has_pattern(self,phns):
        return (True in [esp.has_pattern(p,phns) for p in self.patterns])

    def get_pattern_times(self,phns,phn_times,s,
                     template_length=32):
        feature_start, \
            feature_step, num_features =\
            esp._get_feature_label_times(s,
                                     self.num_window_samples,
                                     self.num_window_step_samples)
        feature_labels, \
            feature_label_transitions \
            = esp._get_labels(phn_times,
                          phns,
                          feature_start,
                          feature_step,
                          num_features,
                          self.sample_rate)
        pattern_times = esp.get_pattern_times(self.patterns,
                                              phns,
                                              feature_label_transitions))
        return pattern_times

    def get_patterns_specs(self,S,phns,phn_times,s,
                           offset=3):
        first_window_s_avg, window_s_avg_step, num_windows = esp._get_spectrogram_label_times(s,
                                     self.num_window_samples,
                                     self.num_window_step_samples)
        spec_labels, \
            spec_label_transitions \
            = esp._get_labels(phn_times,
                          phns,
                          first_window_s_avg,
                          window_s_avg_step,
                          num_windows,
                          self.sample_rate)
        pattern_times = esp.get_pattern_times(self.patterns,
                                              phns,
                                              spec_label_transitions)
        return [S[:,pattern_time[0]-offset:pattern_time[1]+1+offset]\
                        for pattern_time in pattern_times]
            
        
    def get_patterns(self,E,phns,phn_times,s,
                     context=False,template_length=32):
        """
        Parameters
        ----------
        E: array
            edgemap features
        phns: array
            strings representing the labels for the utterance
        phn_times: array
            times when the
        """
        feature_start, \
            feature_step, num_features =\
            esp._get_feature_label_times(s,
                                     self.num_window_samples,
                                     self.num_window_step_samples)
        feature_labels, \
            feature_label_transitions \
            = esp._get_labels(phn_times,
                          phns,
                          feature_start,
                          feature_step,
                          num_features,
                          self.sample_rate)
        pattern_times = esp.get_pattern_times(self.patterns,
                                              phns,
                                              feature_label_transitions)
        if context:
            return [E[:,max(pattern_time[0]-template_length/3,0):\
                            min(pattern_time[0]+template_length+1+ (template_length)/3,E.shape[1])]\
                        for pattern_time in pattern_times]
        else:
            return [E[:,pattern_time[0]:pattern_time[1]+1]\
                        for pattern_time in pattern_times]
         
    def get_unsmoothed_spec(self,s,
                            freq_cutoff=None):
        """
        s: array
            gets the unsmoothed spectrogram
        """
        if freq_cutoff is None:
            return esp.get_log_spectrogram(s,self.sample_rate,
                                   self.num_window_samples,
                                   self.num_window_step_samples,
                                   self.fft_length
                                   )
        else:
            S, freq_idx = esp.get_log_spectrogram(s,self.sample_rate,
                                   self.num_window_samples,
                                   self.num_window_step_samples,
                                   self.fft_length,
                                   return_freqs=True)
            return S[freq_idx<freq_cutoff,:]

    def get_processed_spec(self,s,freq_cutoff=None,
                           fft_length=None,
                           num_window_step_samples = None,
                           num_window_samples=None,
                           kernel_length=None
                           
                           ):
        """
        Wrapper to edge_signal_proc.get_spectrogram_features
        Computes the spectrogram and uses the smoothing
        and frequency cutoff implied by the initial parameter
        settings for the experiment
        """
        if freq_cutoff is None:
            freq_cutoff=self.freq_cutoff
        if fft_length is None:
            fft_length=self.fft_length
        if num_window_step_samples is None:
            num_window_step_samples = self.num_window_step_samples
        if num_window_samples is None:
            num_window_samples=self.num_window_samples
        if kernel_length is None:
            kernel_length=self.kernel_length
        return esp.get_spectrogram_features(s,self.sample_rate,
                                        num_window_samples,
                                        num_window_step_samples,fft_length,
                             freq_cutoff,kernel_length)
        

    def get_mel_spec(self,s,nbands=40,
                     freq_cutoff=None):
        """
        s: array
            gets the unsmoothed spectrogram
        """
        return esp.get_mel_spec(s,self.sample_rate,
                                   self.num_window_samples,
                                   self.num_window_step_samples,
                                   self.fft_length,
                                    nbands
                                   )

    
        
    def get_pattern_parts(self,E,phns,phn_times,s):
        """
        Parameters
        ----------
        E: array
            edgemap features
        phns: array
            strings representing the labels for the utterance
        phn_times: array
            times when the
        """
        feature_start, \
            feature_step, num_features =\
            esp._get_feature_label_times(s,
                                     self.num_window_samples,
                                     self.num_window_step_samples)
        feature_labels, \
            feature_label_transitions \
            = esp._get_labels(phn_times,
                          phns,
                          feature_start,
                          feature_step,
                          num_features,
                          self.sample_rate)
        pattern_part_times = []
        for pattern in self.patterns: pattern_part_times.append(
            esp.get_pattern_part_times(pattern,
                                       phns,
                                       feature_label_transitions))
        return [[E[:,phn_time[0]:phn_time[1]+1] for phn_time in part_time]\
                    for part_time in pattern_part_times]
         

    
    def get_pattern_bgds(self,E,phns,phn_times,s,bg_len,
                         context=False,template_length=33):
        """
        Parameters
        ----------
        E: array
            edgemap features
        phns: array
            strings representing the labels for the utterance
        phn_times: array
            times when the
        """
        feature_start, \
            feature_step, num_features =\
            esp._get_feature_label_times(s,
                                     self.num_window_samples,
                                     self.num_window_step_samples)
        feature_labels, \
            feature_label_transitions \
            = esp._get_labels(phn_times,
                          phns,
                          feature_start,
                          feature_step,
                          num_features,
                          self.sample_rate)
        pattern_times = esp.get_pattern_times(self.patterns,
                                              phns,
                                              feature_label_transitions)
        if context:
            return [E[:,max(pattern_time[0]-template_length/4-bg_len,0):\
                        min(pattern_time[0]+template_length/4+1,E.shape[0])]\
                    for pattern_time in pattern_times]
        else:
            return [E[:,max(pattern_time[0]-bg_len,0):\
                            pattern_time[0]]\
                        for pattern_time in pattern_times]

    def get_pattern_fronts_backs(self,E,phns,phn_times,s,bg_len,
                                 context=False,part_radius=5):
        """
        Parameters
        ----------
        E: array
            edgemap features
        phns: array
            strings representing the labels for the utterance
        phn_times: array
            times when the
        """
        feature_start, \
            feature_step, num_features =\
            esp._get_feature_label_times(s,
                                     self.num_window_samples,
                                     self.num_window_step_samples)
        feature_labels, \
            feature_label_transitions \
            = esp._get_labels(phn_times,
                          phns,
                          feature_start,
                          feature_step,
                          num_features,
                          self.sample_rate)
        pattern_times = esp.get_pattern_times(self.patterns,
                                              phns,
                                              feature_label_transitions)
        if context:
            return [(E[:,max(pattern_time[0]-part_radius,0):\
                             min(pattern_time[0]+part_radius,E.shape[0])],\
                         E[:,max(pattern_time[1]-part_radius,0):\
                                 min(pattern_time[1]+part_radius,E.shape[0])],\
                    pattern_time) for pattern_time in pattern_times]
        else:
            return [(E[:,max(pattern_time[0]-part_radius,0):\
                             min(pattern_time[0]+part_radius,E.shape[1])],\
                         E[:,max(pattern_time[1]-part_radius,0):\
                                 min(pattern_time[1]+part_radius,E.shape[1])],\
                        pattern_time,part_radius) for pattern_time in pattern_times]

    def get_detection_scores_slow(self,E,template,bgd_length,mean_background,
                                  edge_feature_row_breaks,
                                  edge_orientations,
                                  abst_threshold=-np.inf *np.ones(8),
                                  spread_length=3):
        template_height,template_length = template.shape
        num_detections = E.shape[1]-template_length+1
        E_background, estimated_background_idx = self._get_E_background(E,num_detections,bgd_length, mean_background,
                                                                        edge_feature_row_breaks,
                                                                        edge_orientations,
                                                                        abst_threshold=abst_threshold,
                                                                        spread_length=spread_length)
        Ps = np.zeros(num_detections)
        Cs = np.zeros(num_detections)
        for frame_idx in xrange(num_detections):
            E_segment = E[:,frame_idx:frame_idx+template_length].copy()
            esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,abst_threshold=abst_threshold)
            esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=spread_length)
            Ps[frame_idx],Cs[frame_idx] = tt.score_template_background_section(template,E_background[:,frame_idx],E_segment)
        return Ps,Cs

    def get_detection_scores_mix(self,E,bm,bgd_length,mean_background,
                                  edge_feature_row_breaks,
                                  edge_orientations,
                                  abst_threshold=-np.inf *np.ones(8),
                                  spread_length=3):
        """  Do detection for mixture models

        Parameters:
        ===========
        bm - bernoulli mixture model
        """
        mix_scores = []
        for mix_id in range(bm.num_mix):
            mix_Ps,mix_Cs = self.get_detection_scores_slow(E,bm.templates[mix_id],bgd_length,mean_background,
                                  edge_feature_row_breaks,
                                  edge_orientations,
                                  abst_threshold=-np.inf *np.ones(8),
                                  spread_length=3)
            mix_scores.append( mix_Ps+mix_Cs + np.log( bm.weights[mix_id]))
        mix_scores = np.array(mix_scores)
        return np.amax(mix_scores,axis=0)            


    def get_detection_scores(self,E,template,bgd_length, mean_background,
                             edge_feature_row_breaks,
                             edge_orientations,
                             abst_threshold=-np.inf *np.ones(8),
                             spread_length=3):
        """ gets the detection scores for later processing
        
        """
        template_height,template_length = template.shape
        num_detections = E.shape[1]-template_length+1
        E_background, estimated_background_idx = self._get_E_background(E,num_detections,bgd_length, mean_background,
                                                                        edge_feature_row_breaks,
                                                                        edge_orientations,
                                                                        abst_threshold=abst_threshold,
                                                                        spread_length=spread_length)
        E_stack= self._get_E_stack(E,template.shape,num_detections)
        print "E_stack has shape",E_stack.shape
        # perform thresholding for the stack of E features
        for frame_idx in xrange(E_stack.shape[1]):
            E_segment = E_stack[:,frame_idx].reshape(template_height,template_length)
            esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,abst_threshold=abst_threshold)
            esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=spread_length)
        bg_stack = np.tile(E_background,(template.shape[1],1))
        T_stack = np.tile(template.transpose().reshape(np.prod(template.shape),1),
                          (1,num_detections))
        T_inv = 1 - T_stack
        bg_inv = 1- bg_stack
        C_exp_inv_long = T_inv/bg_inv
        C = np.log(C_exp_inv_long).sum(0)
        return (E_stack*np.log(T_stack/bg_stack /\
                                   C_exp_inv_long)).sum(0),\
                                   C
        

    def _get_E_stack(self,E,T_shape,num_detections):
        """ Makes sliding windows simpler to implement
        """
        T_size = int(np.prod(T_shape))
        E_stack_cols = np.tile(np.arange(int(T_size)).reshape(T_size,1),
                         (1,num_detections))
        E_stack_cols = E_stack_cols/T_shape[0]
        E_stack_cols = E_stack_cols + np.arange(int(num_detections))
        
        E_stack_rows = np.tile(np.arange(int(T_shape[0])).reshape(T_shape[0],1),
                               (T_shape[1],num_detections))
        return E[E_stack_rows,E_stack_cols].copy()

        
        
        
        
    def _get_E_background(self,E,num_detections,bgd_length,mean_background,
                          edge_feature_row_breaks,
                          edge_orientations,
                          abst_threshold=-np.inf*np.ones(8),
                          spread_length=3):
        """ Get the background for the whole utterance
        background is computed over a window of length
        bgd_length, and then that's conidered background
        for the following frame in the utterance
        Parameters
        ----------
        E:
            array with speech features
        template_length:
            int for length of the template
        bgd_length:
            int for length of background to sample
        mean_backround:
            1-d array with the mean background

        Returns
        -------
        E_bgd:
            2-d array with the background for all possible
            detections
        bgd_idx:
            1-d array
        """
        # find detection points
        num_bgds = min(E.shape[1]-bgd_length-1,num_detections)
        E_bgd = np.zeros((E.shape[0],num_detections))
        E_bgd[:,:bgd_length+1] = np.tile(mean_background.reshape(E.shape[0],1),
                                     (1,bgd_length+1))
        for d in xrange(num_bgds-bgd_length):
            cur_bgd = E[:,d:d+bgd_length].copy()
            E_bgd[:,d+bgd_length] = self._get_bgd(cur_bgd,
                                                    edge_feature_row_breaks,
                                                    edge_orientations,
                                                  abst_threshold=abst_threshold,
                                                  spread_length=spread_length)
        return E_bgd,np.arange(num_bgds) +bgd_length+1


    def _get_bgd(self,E_bgd_window,
                 edge_feature_row_breaks,
                 edge_orientations,
                 abst_threshold=-np.inf*np.ones(8),
                 spread_length=3):
        esp.threshold_edgemap(E_bgd_window,.30,edge_feature_row_breaks,
                              abst_threshold=abst_threshold)
        esp.spread_edgemap(E_bgd_window,edge_feature_row_breaks,edge_orientations,
                           spread_length=spread_length)
        E_bgd_window = np.mean(E_bgd_window,axis=1)
        E_bgd_window = np.maximum(np.minimum(E_bgd_window,.4),.1)
        return E_bgd_window

            
            
            
        
    
    def get_patterns_negative(self,E,phns,phn_times,s,length):
        """ Returns a negative example for the given pattern
        Parameters
        ----------
        E: array
            edgemap features
        phns: array
            strings representing the labels for the utterance
        phn_times: array
            times when the
        """
        feature_start, \
            feature_step, num_features =\
            esp._get_feature_label_times(s,
                                     self.num_window_samples,
                                     self.num_window_step_samples)
        feature_labels, \
            feature_label_transitions \
            = esp._get_labels(phn_times,
                          phns,
                          feature_start,
                          feature_step,
                          num_features,
                          self.sample_rate)
        pattern_times = esp.get_pattern_times(self.patterns,
                                              phns,
                                              feature_label_transitions)
        end_time = E.shape[1]-length-3
        start_time = length+3
        neg_pattern_time = np.random.randint(start_time,end_time)
        # make sure that its not close to the start time of a patter
        while self._within_pattern_tolerance(neg_pattern_time,
                                        pattern_times,
                                        length):
            neg_pattern_time = np.random.randint(start_time,end_time)
        return E[:,neg_pattern_time:neg_pattern_time+length].copy(), E[:,max(neg_pattern_time-length,0):neg_pattern_time].copy()
         
    def _within_pattern_tolerance(self,neg_pattern_time,
                                  pattern_times,length):
        for pattern_time in pattern_times:
            if (neg_pattern_time >= pattern_time[0] -length/2) and \
                    (neg_pattern_time <= pattern_time[0]+length/2):
                return True
        return False


class Experiment_Iterator(Experiment):
    def __init__(self,base_exp,patterns=None,
                 data_paths=None, 
                 sample_rate=16000,freq_cutoff=3000,
                 num_window_samples=320,
                 num_window_step_samples=80,
                 fft_length=512,
                 data_dir='',kernel_length=7,
                 spread_length=None,
                 abst_threshold=None,
                 bg_len=None,
                 use_mel=False,
                 offset = 3):
        # This says whether to use the mel computed spectrogram or the standard spectrogram
        self.offset = offset
        self.use_mel=use_mel
        self.base_exp = base_exp
        if abst_threshold:
            self.abst_threshold = abst_threshold
        else:
            self.abst_threshold = base_exp.abst_threshold
        if spread_length:
            self.spread_length = spread_length
        else:
            self.spread_length = base_exp.spread_length
        if bg_len:
            self.bg_len = bg_len
        else:
            self.bg_len = base_exp.bg_len
        if kernel_length:
            self.kernel_length = kernel_length
        else:
            self.kernel_length = base_exp.kernel_length        
        if patterns:
            self.patterns = patterns
        else:
            self.patterns = base_exp.patterns
        self.data_dir = base_exp.data_dir
        # make sure that the data dir ends with /
        self.sample_rate=base_exp.sample_rate
        self.freq_cutoff = base_exp.freq_cutoff
        self.num_window_samples = base_exp.num_window_samples
        self.num_window_step_samples = base_exp.num_window_step_samples
        self.fft_length = base_exp.fft_length
        # assume that the files were generated intelligently so there is a correspondence between the labels
        if data_paths:
            self.paths = data_paths
        else:
            self.paths =  base_exp.paths   
        self.num_data = len(self.paths)    
        # file is not currently being read so we begin with a -1 pointer
        self.cur_data_pointer = -1

    def count_num_positive(self):
        self.num_positives = 0
        for cur_example in xrange(self.num_data):
            if self.has_pattern(self.get_phns(cur_example)):
                self.num_positives+=1
        return self.num_positives

    def get_phn_set(self):
        self.phn_set = set()
        for cur_example in xrange(self.num_data):
            self.phn_set.update(self.get_phns(cur_example))
        return self.phn_set

    def get_diphone_counts(self):
        # get dictionary for counts
        self.diphone_counts = {}
        for phn1 in self.phn_set:
            for phn2 in self.phn_set:
                self.diphone_counts[(phn1,phn2)] = 0
        for cur_example in xrange(self.num_data):
            phns = self.get_phns(cur_example)
            for phn_id in xrange(len(phns)-1):
                self.diphone_counts[(phns[phn_id-1],
                                     phns[phn_id])] += 1
        return self.diphone_counts
                

    
    def next(self,wait_for_positive_example=False,
             compute_S=True,
             compute_patterns_specs=False,
             compute_E=True,
             compute_patterns=False, compute_patterns_context=False,
             compute_bgds=False,
             compute_pattern_times=False,
             compute_pattern_parts=False,
             max_template_length = 40):
        """
        Processes the next speech utterance, or the next speech
        utterance that is a positive example.

        Sets the variables:
        self.E
        self.s
        self.phns
        self.phn_times
        self.patterns
        self.pattern_contexts

        Parameters:
        ------------
        wait_for_positive_example: bool
            cycle through the paths until one comes across a positive example

        compute_patterns: bool
            whether to get a list of pattern occurrences from the label set

        compute_patterns_context: bool
            whether to get the patterns with their surrounding context in a list
            for easier processing later

        compute_bgds: bool
            
        """
        if self.cur_data_pointer >= self.num_data-1:
            print "Reached end of data use reset method"
            return False
        else:
            self.cur_data_pointer +=1
        self.phns = self.get_phns(self.cur_data_pointer)
        if wait_for_positive_example:
            cur_data_pointer = self.cur_data_pointer
            no_positives = True
            for i in xrange(1,self.num_data-cur_data_pointer): 
                self.cur_data_pointer = cur_data_pointer + i
                self.phns = self.get_phns(self.cur_data_pointer)
                if self.has_pattern(self.phns):
                    no_positives = False
                    break
            if no_positives:
                print "Error: no positive examples left"
                return False
        self.s = self.get_s(self.cur_data_pointer)
        self.phn_times = self.get_phn_times(self.cur_data_pointer)
        if compute_S:
            self.S = self.get_processed_spec(self.s)
        if compute_patterns_specs:
            self.patterns_specs = self.get_patterns_specs(self.S,
                                                          self.phns,self.phn_times,self.s,
                                                          offset=self.offset)
        if compute_E:
            E,edge_feature_row_breaks,\
                edge_orientations= self.get_edgemap_no_threshold(self.s)
            self.E =E
            self.edge_feature_row_breaks = edge_feature_row_breaks
            self.edge_orientations = edge_orientations
        self.feature_start, \
            self.feature_step, self.num_features =\
            esp._get_feature_label_times(self.s,
                                         self.num_window_samples,
                                         self.num_window_step_samples)
        self.feature_labels, \
            self.feature_label_transitions \
            = esp._get_labels(self.phn_times,
                              self.phns,
                              self.feature_start,
                              self.feature_step,
                              self.num_features,
                              self.sample_rate)

        # select the object
        if compute_patterns:
            self.patterns = self.get_patterns(self.E,self.phns,self.phn_times,self.s)
        if compute_patterns_context:
            self.patterns_context = self.get_patterns(self.E,self.phns,self.phn_times,self.s,context=True,template_length=max_template_length)
        if compute_bgds:
            self.bgds = self.get_pattern_bgds(self.E,self.phns,self.phn_times,self.s,self.bg_len)
        if compute_pattern_times:
            self.pattern_times = self.get_pattern_times(
                                                        self.phns,
                                                        self.phn_times,
                                                        self.s,template_length=32)
        if compute_pattern_parts:
            self.pattern_parts = self.get_pattern_parts(self.E,self.phns,self.phn_times,self.s)
        return True

    def reset_exp(self):
        """
        Reset the data pointer so we start with the first data point
        changes the internal variable self.cur_data_pointer
        """
        self.cur_data_pointer = -1


class AverageBackground:
    def __init__(self):
        self.num_frames = 0
        self.processed_frames = False
    # Method to add frames
    def add_frames(self,E,edge_feature_row_breaks,
                   edge_orientations,abst_threshold):
        new_E = E.copy()
        esp.threshold_edgemap(new_E,.30,edge_feature_row_breaks,abst_threshold=abst_threshold)
        esp.spread_edgemap(new_E,edge_feature_row_breaks,edge_orientations,spread_length=3)
        if not self.processed_frames:
            self.E = np.mean(new_E,axis=1)
            self.processed_frames = True
        else:
            self.E = (self.E * self.num_frames + np.sum(new_E,axis=1))/(self.num_frames+new_E.shape[1])
        self.num_frames += new_E.shape[1]
        


def get_exp_iterator(base_experiment,train_percent=.7):
    last_train_idx = int(base_experiment.num_data*.7)
    return Experiment_Iterator(base_experiment,data_paths=base_experiment.paths[:last_train_idx]),Experiment_Iterator(base_experiment,data_paths=base_experiment.paths[last_train_idx:])


class SlidingWindowJ0:
    def __init__(self,E,template,edge_feature_row_breaks,
                 edge_orientations,spread_length=3,
                 abst_threshold=.0001 * np.ones(8),
                 edge_quantile=.3,
                 window_length=None,
                 j0_threshold=.75,
                 quantile=.75,use_quantile_threshold=True,
                 pattern_times=None,
                 feature_label_transitions=None,
                 phns=None):
        """
        Begins with a mask for the j0 statistics and
        will compute it for a sliding window given a 
        
        Parameters:
        -----------
        E: ndarray
            raw features for edges before any thresholding
            and spreading have been performed

        template: ndarray
            template for doing detection
        
        Optional Parameters:
        --------------------
        window_length=None
            Can be set to an integer that will be the sliding
            window length

        j0_threshold=.75
            Threshold for which edges will be used in the 
            mask for computing j0

        quantile=.75
            Which quantil to use for computing which
            edge features are in the j0 computation

        use_quantile_threshold=True
            whether or not to use a quantile in setting the template
        """
        # get the length of the window for the sliding window
        self.edge_feature_row_breaks = edge_feature_row_breaks
        self.edge_orientations = edge_orientations
        self.spread_length = spread_length
        self.abst_threshold = abst_threshold
        self.edge_quantile = edge_quantile
        if window_length:
            self.window_length = window_length
        else:
            self.window_length = template.shape[1]
        self.template_length = template.shape[1]
        # get the list of indices to use in the J0
        # self.mask has been set
        self.get_mask(template,j0_threshold,
                 quantile,use_quantile_threshold)
        self.num_detections = E.shape[1] - self.window_length
        self.j0_scores = np.empty(self.num_detections)
        self.j0_maxima = []
        self.cur_window_id = -1
        if pattern_times:
            self.pattern_times = pattern_times
            # array to put the labeled_positives data
            self.labeled_positives = []
            self.false_alarms = []
        else:
            self.pattern_times = None
        if feature_label_transitions:
            self.feature_label_transitions = feature_label_transitions
            self.do_label_window=True
        else:
            self.feature_label_transitions = None
            self.do_label_window=False
        if phns:
            self.phns = phns
        else:
            self.phns = None



    def get_mask(self,template,j0_threshold,
                 quantile,use_quantile_threshold):
        if use_quantile_threshold:
            self.template_mask_threshold = max(np.sort(template.ravel())[int(np.prod(template.shape)*quantile)],
                                              j0_threshold)
        else:
            self.template_mask_threshold = j0_threshold
        self.mask = template >= self.template_mask_threshold
        
    def next(self):
        if self.cur_window_id < self.num_detections-1:
            self.cur_window_id+=1
        else:
            print "Error: End of utterance reached, reset window"
            if self.pattern_times:
                self.compute_roc()                    
            return 
        self.cur_window = self.E[:,self.cur_window_id\
                                  :self.cur_window_id+\
                                  self.window_length].copy()
        esp.threshold_edgemap(self.cur_window,self.edge_quantile,
                              self.edge_feature_row_breaks,
                              abst_threshold=self.abst_threshold)
        esp.spread_edgemap(self.cur_window,
                           self.edge_feature_row_breaks,
                           self.edge_orientations,
                           spread_length=self.spread_length)
        self.j0_scores[self.cur_window_id] =\
            np.sum(cur_window[self.mask])
        if self.cur_window_id >=2:
            if self.j0_scores[self.cur_window_id-1]>\
                    max(self.j0_scores[self.cur_window_id],
                        self.j0_scores[self.cur_window_id-2]):
                self.j0_maxima.append(cur_window_id-1)
        
    def reset(self):
        self.cur_window = -1
        

    def compute_roc(self):
        print "Computing the ROC for the j0"
        self.j0_maxima_array = np.array(self.j0_maxima)
        # Find the true positives and false negatives
        j0_maxima_array_bool = np.empty(self.num_detections,dtype=bool)
        j0_maxima_array_bool[:] = False
        j0_maxima_array_bool[self.j0_maxima_array] = True
        for p_id in xrange(len(pattern_times)):
            pattern = pattern_times[p_id]
            positives_array = np.empty(self.num_detections,
                                       dtype=bool)
            positives_array[:] = False
            positives_array[pattern[0]- self.window_length/3:pattern[0]+self.window_length/3] = True
            positives_array = np.logical_and(positives_array,
                                                     j0_maxima_array_bool)
            if np.any(positives_array):
                pos_scores = self.j0_scores[positives_array]
                pos_ex = {"score": np.max(pos_scores),
                          "position": pattern[0] - np.argmax(pos_scores)-pos_scores.shape[0]/2,
                          "pattern_length":pattern[1]-pattern[0],}
                if self.do_label_window:
                    pos_phn_id = np.sum(self.feature_label_transitions <= pattern[0])
                    pos_ex["label_window"] = phns[pos_phn_id-1:pos_phn_id+len(self.pattern)+1]                
                self.labeled_positives.append(pos_ex)
            else:
                print "Error: no positive peaks"
                return
            # we get rid of all the possible negatives here
            j0_maxima_array_bool[pattern[0]- self.window_length/3\
                                     :pattern[0]+self.window_length/3]\
                                     = False
        # now check for false alarms
        new_false_alarms = np.arange(self.num_detections)[j0_maxima_array_bool]
        for fa in new_false_alarms:
            fa_dict = {"score": self.j0_scores[fa]}
            if self.do_label_window:
                phn_id = np.sum(self.feature_label_transitions <= fa)
                pos_ex["label_window"] = phns[phn_id-1:phn_id+len(self.pattern)+1]                
            self.false_alarms.append(fa_dict)
            
                
