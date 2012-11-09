import numpy as np
import template_speech_rec.get_train_data as gtrd
import template_speech_rec.estimate_template as et
import template_speech_rec.bernoulli_mixture as bm
import template_speech_rec.roc_functions as rf
import cPickle,os, pickle,collections

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

SVMResult = collections.namedtuple('SVMResult',
                                   ('num_mix'
                                    +' mix_component'
                                    +' C'
                                    +' W'
                                    +' b'
                                    +' roc_curve'
                                    +' total_error_rate'))
                                    

num_mix = 9
# load in templates
outfile = np.load('aar1_templates_%d.npz' % num_mix)
E_templates = outfile['arr_0']
del outfile

# also want to load in the svm results
outfile = open('aar_svm_result_tuples.pkl','rb')
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


outfile = np.load('data/aar_training_false_pos_%d.npz' % num_mix)
false_pos_templates_avgs = ()
for i in xrange(9):
    false_pos_templates_avgs += (
        np.clip(np.mean(outfile['arr_%d' %i],0),.01,.99),)

del outfile


#
# convert W templates
#
def convert_to_neg_template(P,W):
    return 1./(1. + (1./P -1)*np.exp(W))

def convert_to_pos_template(Q,W):
    return 1./(1. + (1./Q -1)/np.exp(W))



SVM_Neg_Templates = {}
SVM_Pos_Templates = {}
LR_filters ={}
Template_divergences = {}
for i in xrange(9): 
    P = np.clip(E_templates[i],.01,.99)
    QSVM = convert_to_neg_template(P,W_b_templates[i][0])
    bsvm_neg = np.sum((1.-QSVM)/(1. - P))
    SVM_Neg_Templates[i] = (QSVM,bsvm_neg,
                            )
    Q = np.clip(false_pos_templates_avgs[i],.01,.99)
    PSVM = convert_to_pos_template(Q,W_b_templates[i][0])
    bsvm_pos = np.sum((1.-Q)/(1. - PSVM))
    SVM_Pos_Templates[i] = (PSVM,bsvm_pos,
                            )
    LR_filters[i] = (np.log(PSVM)+np.log(1-QSVM)-np.log(1-PSVM) - np.log(QSVM), np.log(P)+np.log(1-Q)-np.log(1-P)-np.log(Q), W_b_templates[i][0],
                     np.log(PSVM)+np.log(1-Q)-np.log(1-PSVM)-np.log(Q),np.log(P)+np.log(1-QSVM)-np.log(1-P)-np.log(QSVM))
    Template_divergences[i] = (P * (np.log(P)-np.log(PSVM)),
                               Q * (np.log(Q) - np.log(QSVM)))

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

    
import matplotlib.cm as cm

for mix_component in xrange(num_mix):
    for edge_feature in xrange(8):
        plt.figure()
        plt.clf()
        # plt.title('Template estimation comparison')        
        plt.subplot(2,2,1)
        # plt.title('Template Trained on True Positive Examples Component %d, num_mix=%d, edge_type=%d'%(mix_component,num_mix,edge_feature))
        plt.imshow(E_templates[mix_component][:,:,edge_feature].T[::-1],
                   interpolation='nearest',
                   vmin=0,vmax=1)
        plt.axis('off')
        plt.subplot(2,2,2)
        # plt.title('Template Trained on False Positive Examples Component %d, num_mix=%d, edge_type=%d'%(mix_component,num_mix,edge_feature))
        plt.imshow(false_pos_templates_avgs[mix_component][:,:,edge_feature].T[::-1],
                   interpolation='nearest',
                   vmin=0,vmax=1)
        plt.axis('off')
        plt.subplot(2,2,3)
        # plt.title('SVM implied Neg Template Component %d, num_mix=%d, edge_type=%d'%(mix_component,num_mix,edge_feature))
        plt.imshow(SVM_Neg_Templates[mix_component][0][:,:,edge_feature].T[::-1],
                   interpolation='nearest',
                   vmin=0,vmax=1)
        plt.axis('off')
        plt.subplot(2,2,4)
        # plt.title('SVM implied Pos Template Component %d, num_mix=%d, edge_type=%d'%(mix_component,num_mix,edge_feature))
        plt.imshow(SVM_Pos_Templates[mix_component][0][:,:,edge_feature].T[::-1],
                   interpolation='nearest',
                   vmin=0,vmax=1)
        plt.axis('off')
        plt.savefig('../13/aar_template_SVMneg_trueneg_%d_%d.png' % (mix_component,edge_feature))
        plt.close('all')
        # need to get normalized figures
        minval = np.min(tuple(t[:,:,edge_feature] for t in LR_filters[mix_component]))
        maxval = np.max(tuple(t[:,:,edge_feature] for t in LR_filters[mix_component]))
        plt.figure()
        plt.clf()
        # plt.colorbar()
        # plt.title('Linear Filter Comparison')
        plt.subplot(3,2,1)
        # plt.title('Likelihood Ratio SVM implied templates %d, num_mix=%d, edge_type=%d'%(mix_component,num_mix,edge_feature))
        plt.imshow(LR_filters[mix_component][0][:,:,edge_feature].T[::-1],
                   interpolation='nearest',
                   vmin=minval,vmax=maxval
                   )
        plt.axis('off')
        plt.subplot(3,2,2)
        # plt.title('Likelihood Ratio Filter Estimated Templates %d, num_mix=%d, edge_type=%d'%(mix_component,num_mix,edge_feature))
        plt.imshow(LR_filters[mix_component][1][:,:,edge_feature].T[::-1],
                   interpolation='nearest',
                   vmin=minval,vmax=maxval
                   )
        plt.axis('off')
        plt.subplot(3,2,3)
        # plt.title('SVM filter %d, num_mix=%d, edge_type=%d'%(mix_component,num_mix,edge_feature))
        plt.imshow(LR_filters[mix_component][2][:,:,edge_feature].T[::-1],
                   interpolation='nearest',
                   #vmin=minval,vmax=maxval
                   )
        plt.axis('off')
        plt.subplot(3,2,4)
        # plt.title('SVM Implied Positive Filter with Estimated Neg %d, num_mix=%d, edge_type=%d'%(mix_component,num_mix,edge_feature))
        plt.imshow(LR_filters[mix_component][3][:,:,edge_feature].T[::-1],
                   interpolation='nearest',
                   #vmin=minval,vmax=maxval
                   )
        plt.axis('off')
        plt.subplot(3,2,5)
        # plt.title('SVM implied Neg Filter with Estimated Pos %d, num_mix=%d, edge_type=%d'%(mix_component,num_mix,edge_feature))
        plt.imshow(LR_filters[mix_component][4][:,:,edge_feature].T[::-1],
                   interpolation='nearest',
                   #vmin=minval,vmax=maxval
                   )
        plt.axis('off')
        plt.savefig('../13/aar_likelihood_ratio_linear_filters_%d_%d.png' % (mix_component,edge_feature))
        plt.close('all')
        minval = np.min(tuple(t[:,:,edge_feature] for t in Template_divergences[mix_component]))
        maxval = np.max(tuple(t[:,:,edge_feature] for t in Template_divergences[mix_component]))
        plt.figure()
        plt.clf()
        # plt.colorbar()
        # plt.title('Template KL Divergences Comparison')
        plt.subplot(1,2,1)
        # plt.title('KL Divergence Estimated to SVM Implied True Pos Template %d, num_mix=%d, edge_type=%d'%(mix_component,num_mix,edge_feature))
        plt.imshow(Template_divergences[mix_component][0][:,:,edge_feature].T[::-1],
                   interpolation='nearest',
                   vmin=minval,vmax=maxval)
        plt.axis('off')
        plt.subplot(1,2,2)
        # plt.title('KL divergence Estimated to SVM Implied False Pos %d, num_mix=%d, edge_type=%d'%(mix_component,num_mix,edge_feature))
        plt.imshow(Template_divergences[mix_component][1][:,:,edge_feature].T[::-1],
                   interpolation='nearest',
                   vmin=minval,vmax=maxval)
        plt.axis('off')
        plt.savefig('../13/aar_template_divergences_%d_%d.png' % (mix_component,edge_feature))
        plt.close('all')
