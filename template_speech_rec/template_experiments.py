import numpy as np
import edge_signal_proc as esp
import test_template as tt

class Experiment:
    def __init__(self,sample_rate,freq_cutoff,
                 num_window_samples,
                 num_window_step_samples,
                 fft_length,s_files_path_file,
                 phns_files_path_file,
                 phn_times_files_path_file,
                 data_dir,pattern,kernel_length):
        self.kernel_length = kernel_length
        self.pattern = pattern
        self.data_dir = data_dir
        # make sure that the data dir ends with /
        if self.data_dir[-1] != '/':
            self.data_dir += '/'
        self.sample_rate=sample_rate
        self.freq_cutoff = freq_cutoff
        self.num_window_samples = num_window_samples
        self.num_window_step_samples = num_window_step_samples
        self.fft_length = fft_length
        # assume that the files were generated intelligently so there is a correspondence between the labels
        self.s_paths = [s_path for s_path in \
                            open(s_files_path_file,'r').read().split('\n') \
                            if s_path !='']
        self.phns_paths = [phns_path for phns_path in \
                               open(phns_files_path_file,'r').read().split('\n') \
                               if phns_path !='']
        self.phn_times_paths = [phn_times_path for phn_times_path in \
                                    open(phn_times_files_path_file,'r').read().split('\n') \
                                    if phn_times_path != '']
        self.num_data = len(self.s_paths)
        # test the correspondence
        assert (len(self.s_paths) == len(self.phns_paths))
        assert (len(self.phn_times_paths) == len(self.phns_paths))
        for f in range(len(self.s_paths)):
            assert (self.s_paths[f].split('_')[0] ==\
                        self.phns_paths[f].split('_')[0])
            assert (self.phn_times_paths[f].split('_')[0] ==\
                        self.phns_paths[f].split('_')[0])
    

    def get_s(self,idx):
        return np.loadtxt(self.data_dir+self.s_paths[idx])

    def get_phns(self,idx):
        return np.array([phn for phn in \
                    open(self.data_dir+self.phns_paths[idx],'r').read().split('\n') \
                    if phn != ''])

    def get_phn_times(self,idx):
        return np.loadtxt(self.data_dir+self.phn_times_paths[idx])



        
    def get_edgemap_no_threshold(self,s):
        return esp.get_edgemap_no_threshold(s,self.sample_rate,
                                             self.num_window_samples,
                                             self.num_window_step_samples,
                                             self.fft_length,
                                             self.freq_cutoff,
                                             self.kernel_length)
        
    def has_pattern(self,phns):
        return esp.has_pattern(self.pattern,phns)

    def get_pattern_times(self,phns,phn_times,s,
                     context=False,template_length=32):
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
        return esp.get_pattern_times(self.pattern,
                                              phns,
                                              feature_label_transitions)


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
        pattern_times = esp.get_pattern_times(self.pattern,
                                              phns,
                                              feature_label_transitions)
        if context:
            return [E[:,max(pattern_time[0]-template_length/4,0):\
                            min(pattern_time[0]+template_length+1+ (template_length)/4,E.shape[1])]\
                        for pattern_time in pattern_times]
        else:
            return [E[:,pattern_time[0]:pattern_time[1]+1]\
                        for pattern_time in pattern_times]
         
    
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
        pattern_times = esp.get_pattern_times(self.pattern,
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
        pattern_times = esp.get_pattern_times(self.pattern,
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
        pattern_times = esp.get_pattern_times(self.pattern,
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

