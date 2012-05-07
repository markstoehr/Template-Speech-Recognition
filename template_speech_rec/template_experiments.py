import numpy as np
import edge_signal_proc as esp
import test_template as tt
import random




class Experiment:
    def __init__(self,pattern,data_paths_file,bg_len=26,
                 sample_rate=16000,freq_cutoff=3000,
                 num_window_samples=320,
                 num_window_step_samples=80,
                 fft_length=512,
                 data_dir='',kernel_length=7,
                 spread_length=3,
                 abst_threshold=.0001*np.ones(8),
                 do_random=True):
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
        self.bg_len = bg_len
        self.spread_length = spread_length
        self.abst_threshold = abst_threshold
        self.kernel_length = kernel_length
        self.pattern = pattern
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


    def get_s(self,idx):
        return np.loadtxt(self.paths[idx] + '_s.txt')

    def get_phns(self,idx):
        return np.array([phn for phn in \
                    open(self.paths[idx]+'_phns.txt','r').read().split('\n') \
                    if phn != ''])

    def get_phn_times(self,idx):
        return np.loadtxt(self.paths[idx]+'_phn_times.txt')



        
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


class Experiment_Iterator(Experiment):
    def __init__(self,base_exp,pattern=None,
                 data_paths=None, 
                 sample_rate=16000,freq_cutoff=3000,
                 num_window_samples=320,
                 num_window_step_samples=80,
                 fft_length=512,
                 data_dir='',kernel_length=7,
                 spread_length=None,
                 abst_threshold=None,
                 bg_len=None):
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
        if pattern:
            self.pattern = pattern
        else:
            self.pattern = base_exp.pattern
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
    
    def next(self,wait_for_positive_example=False,
             get_patterns=False, get_patterns_context=False,
             get_bgds=False,get_pattern_times=False):
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
        E,edge_feature_row_breaks,\
            edge_orientations= self.get_edgemap_no_threshold(self.s)
        self.E =E
        self.edge_feature_row_breaks = edge_feature_row_breaks
        self.edge_orientations = edge_orientations
        self.phn_times = self.get_phn_times(self.cur_data_pointer)
        # select the object
        if get_patterns:
            self.patterns = self.get_patterns(self.E,self.phns,self.phn_times,self.s)
        if get_patterns_context:
            self.patterns_context = self.get_patterns(self.E,self.phns,self.phn_times,self.s,context=True,template_length=33)
        if get_bgds:
            self.bgds = self.get_pattern_bgds(self.E,self.phns,self.phn_times,self.s,self.bg_len)
        if get_pattern_times:
            self.pattern_times = get_pattern_times(self,
                                                   self.phns,
                                                   self.phn_times,
                                                   self.s,
                                                   context=False,template_length=32)
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
    def __init__(self,E,template,
                 window_length=None,
                 j0_threshold=.75,
                 quantile=.75,use_quantile_threshold=True):
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
        self.j0_scores = np.empty(num_detections)
        self.j0_maxima = []


    def get_mask(self,template,j0_threshold,
                 quantile,use_quantile_threshold):
        if use_quantile_threshold:
            self.template_mask_threshold = np.max(np.sort(template.ravel())[int(self.template_length*quantile)],
                                              j0_threshold)
        else:
            self.template_mask_threshold = j0_threshold
        self.mask = template >= self.template_mask_threshold
        
    def next(self):
        
        
