import numpy as np
import template_speech_rec.get_train_data as gtrd
import template_speech_rec.estimate_template as et
import template_speech_rec.bernoulli_mixture as bm
import template_speech_rec.roc_functions as rf
import cPickle,os, pickle

sp = gtrd.SpectrogramParameters(
    sample_rate=16000,
    num_window_samples=320,
    num_window_step_samples=80,
    fft_length=512,
    kernel_length=7,
    freq_cutoff=3000,
    use_mel=False)

ep = gtrd.EdgemapParameters(block_length=40,
                            spread_length=1,
                            threshold=.7)

# load in templates
outfile = np.load('aar1_templates_%d.npz' % num_mix)
E_templates = outfile['arr_0']
del outfile

# also want to load in the svm results
outfile = np.open('aar_svm_result_tuples.pkl','rb')
svmresult_tuple = pickle.load(outfile)
outfile.close()

# we get the 9 templates
# maximal regularization
# C
min_C = min( s.C for s in svmresult_tuple)
num_mix = max( s.num_mix for s in svmresult_tuple)
W_b_templates = {}
for i, s in enumerate(svmresult_tuple):
    if s.C == min_C and s.num_mix == num_mix:
        W_b_templates[s.mix_component] =(s.W,s.b,s.roc_curve)


#
# convert W templates
#
def convert_to_template(P,W):
    return 1./(1. + (1./P -1)*np.exp(W))

SVM_Neg_Templates = {}
for i in xrange(9): 
    P = np.clip(E_templates[i],.01,.99)
    Q = convert_to_template(P,W_b_templates[i][0])
    b = np.sum((1.-Q)/(1. - P))
    SVM_Neg_Templates[i] = (Q,b,
                            )


for i in xrange(9): 
    for j in np.arange(3,10):
        alpha = 10.**(-j)
        P = np.clip(E_templates[i],alpha,1-alpha)
        Q = convert_to_template(P,W_b_templates[i][0])
        print i, Q.min(), Q.max(), alpha

#
# want to also get some sense of the roc curve
#

#
# also going to get the false positive templates from the clusters
#

outfile = np.load('data/aar_training_false_pos_%d.npz' % num_mix)
false_pos_templates_avgs = ()
for i in xrange(9):
    false_pos_templates_avgs += (
        np.clip(np.mean(outfile['arr_%d' %i],0),.01,.99),)

del outfile
    
import matplotlib.cm as cm

for mix_component in xrange(num_mix):
    for edge_feature in xrange(8):
        plt.figure()
        plt.clf()
        plt.subplot(3,1,1)
        plt.imshow(E_templates[mix_component][:,:,edge_feature].T[::-1],
                   interpolation='nearest',
                   vmin=0,vmax=1,cmap=cm.bone)
        plt.axis('off')
        plt.subplot(3,1,2)
        plt.imshow(SVM_Neg_Templates[mix_component][0][:,:,edge_feature].T[::-1],
                   interpolation='nearest',
                   vmin=0,vmax=1,cmap=cm.bone)
        plt.axis('off')
        plt.subplot(3,1,3)
        plt.imshow(false_pos_templates_avgs[mix_component][:,:,edge_feature].T[::-1],
                   interpolation='nearest',
                   vmin=0,vmax=1,cmap=cm.bone)
        plt.axis('off')
        plt.savefig('../13/aar_template_SVMneg_trueneg_%d_%d.png' % (mix_component,edge_feature))
        plt.close()


