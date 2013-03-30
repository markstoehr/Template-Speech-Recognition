#!/usr/bin/python
import numpy as np
import pylab as pl
from sklearn import svm
import template_speech_rec.get_train_data as gtrd
import template_speech_rec.estimate_template as et
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.bernoulli_mixture as bm
import template_speech_rec.roc_functions as rf
import template_speech_rec.code_parts as cp
import template_speech_rec.spread_waliji_patches as swp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import template_speech_rec.code_parts as code_parts
import pickle,collections,cPickle

sp = gtrd.makeSpectrogramParameters(
    sample_rate=16000,
    num_window_samples=320,
    num_window_step_samples=80,
    fft_length=512,
    kernel_length=7,
    freq_cutoff=3000,
    use_mel=False,
    )

ep = gtrd.makeEdgemapParameters(block_length=40,
                            spread_length=1,
                            threshold=.7)

def getEdgeDistribution(file_indices,
                        hw,
                        file_indices_chunks=20):
    return None

def get_file_indices(file_indices_path,data_path):
    try:
        file_indices=np.load(file_indices_path)
    except:
        file_indices = gtrd.get_data_files_indices(data_path)
        np.save(file_indices_path,file_indices)
    return file_indices

def get_params(args,
               sample_rate=16000,
    num_window_samples=320,
    num_window_step_samples=80,
    fft_length=512,
    kernel_length=7,
    freq_cutoff=3000,
    use_mel=False
               ):
    # we get the basic file paths right here
    # TODO: make this system adaptive
    root_path = '/home/mark/Template-Speech-Recognition/'
    utterances_path = '/home/mark/Template-Speech-Recognition/Data/Train/'
    try:
        file_indices = np.load('data_parts/train_file_indices.npy')
    except:
        file_indices = gtrd.get_data_files_indices(utterances_path)
        np.save('data_parts/train_file_indices.npy',file_indices)

    num_mix_params = [1,2,3,5,7,9]

    test_path = '/home/mark/Template-Speech-Recognition/Data/Test/'
    train_path = '/home/mark/Template-Speech-Recognition/Data/Train/'

    try:
        test_example_lengths =np.load("data_parts/test_example_lengths.npy")
        test_file_indices = np.load("data_parts/test_file_indices.npy")
    except:
        test_file_indices = gtrd.get_data_files_indices(test_path)
        test_example_lengths = gtrd.get_detect_lengths(test_file_indices,test_path)
        np.save("data_parts/test_example_lengths.npy",test_example_lengths)
        np.save("data_parts/test_file_indices.npy",test_file_indices)
        
    try:
        train_example_lengths =np.load("data_parts/train_example_lengths.npy")
        train_file_indices = np.load("data_parts/train_file_indices.npy")
    except:
        train_file_indices = gtrd.get_data_files_indices(train_path)
        train_example_lengths = gtrd.get_detect_lengths(train_file_indices,train_path)
        np.save("data_parts/train_example_lengths.npy",train_example_lengths)
        np.save("data_parts/train_file_indices.npy",train_file_indices)


    return (gtrd.SpectrogramParameters(
            sample_rate=16000,
            num_window_samples=320,
            num_window_step_samples=80,
            fft_length=512,
            kernel_length=7,
            freq_cutoff=3000,
            use_mel=args.use_mel),
            gtrd.makeEdgemapParameters(block_length=args.edgeMapBlockLength,
                                        spread_length=args.edgeMapSpreadLength,
                                        threshold=args.edgeMapThreshold),
            root_path,utterances_path,
            file_indices,num_mix_params,
            test_path,train_path,
            train_example_lengths, train_file_indices,
            test_example_lengths, test_file_indices)

def save_syllable_features_to_data_dir(args,phn_tuple,
                          utterances_path,
                          file_indices,
                          
                         sp,ep,
                          phn_mapping,tag_data_with_syllable_string=False,
                                       save_tag="train",
                          waveform_offset=10,
                                      block_features=False):
    """
    Wrapper function to get all the examples processed
    """
    print "Collecting the data for phn_tuple " + ' '.join('%s' % k for k in phn_tuple)
    syllable_string = '_'.join(p for p in phn_tuple)

    phn_features,avg_bgd=gtrd.get_syllable_features_directory(
        utterances_path,
        file_indices,
        phn_tuple,
        S_config=sp,E_config=ep,offset=0,
        E_verbose=False,return_avg_bgd=True,
        waveform_offset=15,
        phn_mapping=phn_mapping)
    bgd = np.clip(avg_bgd.E,.01,.4)
    np.save('data/bgd.npy',bgd)
    example_mat = gtrd.recover_example_map(phn_features)
    lengths,waveforms  = gtrd.recover_waveforms(phn_features,example_mat)
    if tag_data_with_syllable_string:
        np.savez('data/%s_waveforms_lengths_%s.npz' % (syllable_string,
                                                       save_tag),
                 waveforms=waveforms,
                 lengths=lengths,
                 example_mat=example_mat)
    else:
        np.savez('data/waveforms_lengths_%s.npz' % save_tag,waveforms=waveforms,
                 lengths=lengths,
                 example_mat=example_mat)
    Slengths,Ss  = gtrd.recover_specs(phn_features,example_mat)
    Ss = Ss.astype(np.float32)
    if tag_data_with_syllable_string:
        np.savez('data/%s_Ss_lengths_%s.npz' % (syllable_string,
                                                       save_tag),Ss=Ss,Slengths=Slengths,example_mat=example_mat)
    else:
        np.savez('data/Ss_lengths_%s.npz' % (
                                                       save_tag),Ss=Ss,Slengths=Slengths,example_mat=example_mat)
    Elengths,Es  = gtrd.recover_edgemaps(phn_features,example_mat,bgd=bgd)
    Es = Es.astype(np.uint8)
    if tag_data_with_syllable_string:
        np.savez('data/%s_Es_lengths_%s.npz'% (syllable_string,
                                                       save_tag) ,Es=Es,Elengths=Elengths,example_mat=example_mat)
    else:
        np.savez('data/Es_lengths_%s.npz'% (
                                                       save_tag) ,Es=Es,Elengths=Elengths,example_mat=example_mat)

    if args.doBlockFeatures:
        out = code_parts.code_parts(E.astype(np.uint8),
                                        logParts,logInvParts,args.bernsteinEdgeThreshold)
        max_responses = np.argmax(out,-1)
        Bs = code_parts.spread_patches(max_responses,2,2,out.shape[-1]-1)



def visualize_edge_on_specs(fl_path,E,S):
    plt.close('all')
    plt.figure()
    plt.clf()
    for i in xrange(E.shape[-1]):
        plt.subplot(4,2,i+1)
        plt.imshow(S.T.astype(np.float),aspect=2,origin="lower",alpha=.8)
        plt.imshow(E[:,:,i].T.astype(np.float),cmap=cm.bone,vmin=0,vmax=1,origin="lower left",alpha = .4,aspect=2)
        plt.axis('off')
    plt.savefig(fl_path)
    plt.close('all')

def visualize_bern_on_specs(fl_path,max_responses,S,part_id):
    S_view = S[:max_responses.shape[0],
                :max_responses.shape[1]]
    plt.close('all')
    plt.figure()
    plt.clf()
    plt.imshow(S_view.T.astype(np.float),aspect=3,origin="lower",alpha=.8)
    plt.imshow((max_responses==part_id+1).T.astype(np.float),cmap=cm.bone,vmin=0,vmax=1,origin="lower left",alpha = .4,aspect=3)
    
    plt.axis('off')
    plt.savefig(fl_path)
    plt.close('all')
 
def get_streaming_quantiles(E,quantileSet,blockLength):
    out = np.zeros((len(quantileSet),E.shape[1]-blockLength+1))
    
    for i in xrange(out.shape[1]):
        nz_block = E[:,i:i+blockLength][E[:,i:i+blockLength]>0].ravel()
        len_nz_block =len(nz_block)
        sortedBlock = np.sort(nz_block)
        if len_nz_block ==0 :
            out[:,i]=0
        else:
            out[:,i]=nz_block[ np.clip(quantileSet *len_nz_block,0,len_nz_block-1).astype(int)]
    return out

def get_streaming_heavy_hitters(E,heavyHitterSet,blockLength):
    out = np.zeros((len(quantileSet),E.shape[1]-blockLength+1))
    
    for i in xrange(out.shape[1]):
        nz_block = E[:,i:i+blockLength].sum()
        sum_nz_block =np.sum(nz_block)
        sortedBlock = np.sort(nz_block)
        if len_nz_block ==0 :
            out[:,i]=0
        else:
            out[:,i]=nz_block[ np.clip(quantileSet *len_nz_block,0,len_nz_block-1).astype(int)]
    return out


def visualize_spec_and_quantiles(fpath,E,S,
                                         quantileSet,blockLengthSet):
    
    plt.close('all')
    quantileSet = np.array(quantileSet)
    for i in xrange(8):
        fig = plt.figure()
        plt.figtext(.5,.965,'Comparing spectrogram to edge quantiles for edge type %d' % i )
        num_plots = 1+len(blockLengthSet)
        ax1=plt.subplot(num_plots,1,1)
        ax1.imshow(S.T.astype(np.float),aspect=2,origin="lower",alpha=.8)
        #import pdb; pdb.set_trace()
        ax1.imshow(E[:,:,i],origin="lower left",
                   #cmap=cm.bone,vmin=0,vmax=1,
                   #cmap=cm.binary,
                   alpha = .3,aspect=2)
        ax1.axis('off')
        axjs =[]
        for j in xrange(len(blockLengthSet)):
            axjs.append(plt.subplot(num_plots,1,j+2,sharex=ax1))
            
            quantile_stream =get_streaming_quantiles(E[:,:,i],
                                    quantileSet,
                                    blockLengthSet[j])

            axjs[j].imshow(
            quantile_stream,
            origin="lower left",aspect=2)
            axjs[j].axis('off')

        fig.tight_layout()
        plt.savefig(fpath+'_%d.png'%i)
        plt.close('all')

    
def main(args):
    if args.v:
        print args
        print "Checking value of args.useDefaultParams=",args.useDefaultParams
    if args.useDefaultParams:
        if args.v:
            # retrieving params with function
            print "Using default parameters"
        (sp,
     ep,
     root_path,utterances_path,
     file_indices,num_mix_params,
     test_path,train_path,
     train_example_lengths, train_file_indices,
     test_example_lengths, test_file_indices) = get_params(args)
    else:
        # need to change this to something else
        (sp,
     ep,
     root_path,utterances_path,
     file_indices,num_mix_params,
     test_path,train_path,
     train_example_lengths, train_file_indices,
     test_example_lengths, test_file_indices) =get_params(args)
    file_indices=get_file_indices(args.fileIndicesPath,
                                  args.dataPath)
    if args.limitFileIndices > -1:
        if args.v:
            print "Limiting file indices to length %d" % args.limitFileIndices
        file_indices=file_indices[:args.limitFileIndices]
    elif args.v:
        print "No limit on file indices"

    if args.partsPath != '':
        if args.v:
            print "Loading in parts from %s" % args.partsPath
        EParts=np.clip(np.load(args.partsPath),.01,.99)
        logParts=np.log(EParts).astype(np.float64)
        logInvParts=np.log(1-EParts).astype(np.float64)
        
    if args.printEdgeDistribution:
        edge_distribution=getEdgeDistribution(file_indices,
                                              args.hw,
                                              file_indices_chunks=args.file_indices_chunks)
    elif args.edgeQuantileComparison != '':
        for fl_id, fl in enumerate(file_indices):
            if args.v:
                print fl_id
            utterance = gtrd.makeUtterance(args.dataPath,fl)
            print sp, args.mel_smoothing_kernel
            S = gtrd.get_spectrogram(utterance.s,sp,mel_smoothing_kernel=args.mel_smoothing_kernel)
            E, edge_feature_row_breaks,\
            edge_orientations = esp._edge_map_no_threshold(S.T)
            E2 = np.empty((E.shape[0]/8,E.shape[1],8))
            for i in xrange(8):
                E2[:,:,i] = E[E.shape[0]/8 *i:E.shape[0]/8 *(i+1),:]
            print E2.shape,S.shape
            visualize_spec_and_quantiles('%s_%d' % (args.edgeQuantileComparison,
                                                        fl_id),E2,S,
                                         args.quantileSet,args.blockLengthSet)
    elif args.createBackgroundMixture > 0:
        pass
    elif args.getUnclippedBackground != '':
        # initialize the background
        if args.v:
            print "Initializing average background to be computed over parts"
        avg_bgd = gtrd.AverageBackground()
        for fl_id, fl in enumerate(file_indices):
            if args.v:
                print fl_id
            utterance = gtrd.makeUtterance(args.dataPath,fl)
            print sp, args.mel_smoothing_kernel
            S = gtrd.get_spectrogram(utterance.s,sp,mel_smoothing_kernel=args.mel_smoothing_kernel)
            E = gtrd.get_edge_features(S.T,ep,verbose=False
                                       )
            if args.seeBackgroundEstimatePlots != '':
                visualize_edge_on_specs('%s_%d.png' %(args.seeBackgroundEstimatePlots,
                                                      fl_id),
                                        E,S)
            out = code_parts.code_parts(E.astype(np.uint8),
                                        logParts,logInvParts,args.bernsteinEdgeThreshold)
            max_responses = np.argmax(out,-1)
            if args.bernsteinPreSpreadVisualizeOnSpec != '':
                # cycle over all parts
                for part_id in xrange(logParts.shape[0]):
                    visualize_bern_on_specs('%s_%d_%d.png' % (args.bernsteinPreSpreadVisualizeOnSpec,
                                                              fl_id,
                                                              part_id),max_responses,S,part_id)
            bin_out_map = code_parts.spread_patches(max_responses,2,2,out.shape[-1]-1)
            avg_bgd.add_frames(bin_out_map,time_axis=0)
            
        np.save(args.getUnclippedBackground,avg_bgd.E)
    else:
        pass
                                              


if __name__=="__main__":
    # parsing module
    
    parser = argparse.ArgumentParser(description='Estimate the parts model.')
    parser.add_argument('-v',action='store_true',
                        help="verbose")
    parser.add_argument('-s','--hw', metavar='N', type=int, nargs=2,
                        help='the heights and widths of the parts',
                        default=[5,5])
    parser.add_argument('--dataPath',metavar='Path',type=str,nargs=1,
                        help='path to where the files containing utterances are stored',
                        default='/home/mark/Template-Speech-Recognition/Data/Train/')
    parser.add_argument('--fileIndicesPath',metavar='Path',type=str,nargs=1,
                        help='path where the indices of the speech files arestored',
                        default='/home/mark/Template-Speech-Recognition/Notebook/19/data_parts/trainFileIndices.npy')
    parser.add_argument('--fileIndicesChunks',
                        metavar='N',type=int,nargs=1,
                        help='Chunks of files to be processed at a time',
                        default=20)
    parser.add_argument('--edgeMinMaxCounts',metavar='edgeMinMaxCounts',type=int,nargs=2,
                        help='Minimum and maximum counts for a patch to be considered a patch',
                        default=[30,90])
    parser.add_argument('--printEdgeDistribution',metavar='printEdgeDist',
                        type=bool,help="Part model not estimated just the distribution of edges",
                        default=False)
    parser.add_argument('--partsPath',metavar='Path',
                        type=str,help="Path to the file where the parts are saved or will be saved",
                        default="/home/mark/Template-Speech-Recognition/"
                        + "Development/102012/"
                        + "E_templates.npy")
    parser.add_argument('--getUnclippedBackground',
                        metavar='Path',
                        type=str,help="Whether to run the program"
                        +" where we just check the part distribution, do the"
                        +" estimation if a path is given and nothing otherwise",
                        default='')
    parser.add_argument('--useDefaultParams',
                        metavar='T/F',
                        type=bool,help="Where to use the default get_params"
                        +" function to get the basic parameter set",
                        default=True)
    parser.add_argument('--limitFileIndices',
                        metavar='N',
                        type=int,help="Use only a certain number of file_indices, -1 means no constraint",
                        default=-1)
    parser.add_argument('--seeBackgroundEstimatePlots',metavar='Path',
                        type=str,default='',
                        help='Displays the edge features to visualize what is happening if this is set to some path other than the null string')
    
    # setting the edge map computation parameters
    parser.add_argument('--edgeMapBlockLength',metavar='N',
                        type=int,default=40,
                        help="Block length over which we compute the"
                        + " threshold")
    parser.add_argument('--edgeMapSpreadLength',metavar='N',
                        type=int,default=1,
                        help="Length of spreading performed on the edge map"
                        +" features")
    parser.add_argument('--edgeMapThreshold',metavar='F',
                        type=float,default=.7,
                        help="Quantile to threshold the edge values at for detection")
    parser.add_argument('--bernsteinEdgeThreshold',metavar='N',
                        type=int,default=18,
                        help="Threshold for the edge counts before a part is fit to a location")
    parser.add_argument('--bernsteinPreSpreadVisualizeOnSpec',
                        metavar='Path',
                        type=str,default='',
                        help="If included and not empty this is the path that the graphs are saved to for visualizing the part models")

    parser.add_argument('--phone',
                        metavar="PHONE",
                        type=str,default='',nargs='+',
                        help="Sequence of timit phones for constructing the recognition sequence")
    parser.add_argument('--saveFeaturesToDirectory')
    parser.add_argument('--use_mel',action='store_true',
                        help="whether to use mel features")
    parser.add_argument('--mel_smoothing_kernel',default=-1,type=int,
                        metavar='N',help="The smoothing kernel length, default is -1 so no smoothing is done")
    parser.add_argument('--edgeQuantileComparison',default='',type=str,
                        help="Displays a plot that shows the quantiles as a function of time")
    parser.add_argument('--blockLengthSet',nargs='+',default=[40],type=int,
                        help="Stores different block lengths for the edge thresholding visualization experiments")
    parser.add_argument('--quantileSet',nargs='+',default=[.7],type=float,
                        help="Set of quantiles for visualization of how different quantiles affect edge thresholding")
    parser.add_argument('--createBackgroundMixture',type=int,default=0,
                        help="number of components to be used for making the background mixture.  Default is 0, and then nothing is estimated")
    main(parser.parse_args())
