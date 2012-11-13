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

def iterative_neg_template_estimate(svmW,P,svmC,
                                    max_iter = 1000,
                                    tol=.00001,
                                    verbose=False):
    scaling = 1.
    W = scaling * svmW
    C = scaling * svmC
    lfW = np.zeros(W.shape)
    cur_error = abs(np.sum(scaling*W - lfW))/abs(scaling*W.sum())
    num_iter = 0
    while cur_error > tol and num_iter < max_iter:
        Q = np.clip(convert_to_neg_template(P,scaling*svmW),.0001,1-.0001)
        lfC = np.sum(np.log((1-P)/(1-Q)))
        scaling = lfC/C
        lfW = np.log(P/(1.-P) * (1.-Q)/Q)
        cur_error = abs(np.sum(scaling*W - lfW))/abs(scaling*W.sum())
        if verbose:
            print "lfC=%g\tnorm(lfW)=%g\tsvmC=%g\tcur_error=%g\tscaling=%g" % (lfC,np.linalg.norm(lfW),svmC,cur_error,scaling)
        num_iter += 1
    print "lfC=%g\tnorm(lfW)=%g\tsvmC=%g\tcur_error=%g\tscaling=%g" % (lfC,np.linalg.norm(lfW),svmC,cur_error,scaling)
    return lfW,Q,lfC


iterative_neg_template_estimate(W_b_templates[0][0],E_templates[0],W_b_templates[0][1],
                                    num_iter = 1000,
                                    tol=.00001)

svmW = W_b_templates[1][0].copy()
P = np.clip(E_templates[1].copy(),.0001,1-.0001)
svmC = W_b_templates[1][1].copy()
num_iter = 1000
tol=.00001

scaling = 1.
W = scaling * svmW
C = scaling * svmC
lfW = np.zeros(W.shape)
cur_error = abs(np.sum(scaling*W - lfW))/abs(scaling*W.sum())
while cur_error > tol:
    Q = np.clip(convert_to_neg_template(P,scaling*svmW),.0001,1-.0001)
    lfC = np.sum(np.log((1-P)/(1-Q)))
    scaling = lfC/C
    lfW = np.log(P/(1.-P) * (1.-Q)/Q)
    cur_error = abs(np.sum(scaling*W - lfW))/abs(scaling*W.sum())
    print "lfC=%g\tnorm(lfW)=%g\tsvmC=%g\tcur_error=%g\tscaling=%g" % (lfC,np.linalg.norm(lfW),svmC,cur_error,scaling)

W = scaling * svmW
C = scaling * svmC




def get_best_svm_scaling(lfW,svmW,lfC,svmC):
    return (np.sum(lfW * svmW) + lfC * svmC)/(np.sum(lfW * lfW) + svmC**2)

def convert_to_pos_template(Q,W):
    return 1./(1. + (1./Q -1)/np.exp(W))



SVM_Neg_Templates = {}
SVM_Pos_Templates = {}
LR_filters ={}
Template_divergences = {}
for i in xrange(9): 
    P = np.clip(E_templates[i],.01,.99)
    lfW,QSVM,bsvm_neg = iterative_neg_template_estimate(W_b_templates[i][0],P,W_b_templates[0][1],
                                    max_iter = 500,
                                    tol=.000001,
                                                        verbose=False)
    SVM_Neg_Templates[i] = (QSVM,bsvm_neg,
                            )
#    Q = np.clip(false_pos_templates_avgs[i],.01,.99)



    # LR_filters[i] = (np.log(PSVM)+np.log(1-QSVM)-np.log(1-PSVM) - np.log(QSVM), np.log(P)+np.log(1-Q)-np.log(1-P)-np.log(Q), W_b_templates[i][0],
    #                  np.log(PSVM)+np.log(1-Q)-np.log(1-PSVM)-np.log(Q),np.log(P)+np.log(1-QSVM)-np.log(1-P)-np.log(QSVM))
    # Template_divergences[i] = (P * (np.log(P)-np.log(PSVM)),
    #                            Q * (np.log(Q) - np.log(QSVM)))

for i in xrange(9): 
    for j in np.arange(3,10):
        alpha = 10.**(-j)
        P = np.clip(E_templates[i],alpha,1-alpha)
        Q = convert_to_template(P,W_b_templates[i][0])
        print i, Q.min(), Q.max(), alpha

#
# want to also get some sense of the roc curve
#


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
        # plt.title('SVM implied Neg Template Component %d, num_mix=%d, edge_type=%d'%(mix_component,num_mix,edge_feature))
        plt.imshow(np.abs(SVM_Neg_Templates[mix_component][0][:,:,edge_feature].T[::-1]-E_templates[mix_component][:,:,edge_feature].T[::-1]),
                   interpolation='nearest',
                   vmin=0,vmax=1)
        plt.axis('off')
        plt.savefig('../13/aarIterative_template_SVMneg_trueneg_%d_%d.png' % (mix_component,edge_feature))


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
