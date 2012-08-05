# This file is here just to figure out how to properly visualize the templates
import matplotlib.pyplot as plt
import numpy as np

template_path = '/home/mark/Template-Speech-Recognition/Experiments/080212/'

p_template = np.load(template_path+'2_0_p_templates.npy')

plt.imshow(p_template[0],interpolation='nearest')
plt.show()


plt.imshow(p_template[0],interpolation='nearest',
           aspect=.2)
plt.show()

p_template = np.load(template_path+'10_0_p_templates.npy')

n=10
num_plots = n/2+1
num_cols = 2
fig = plt.figure()
for i in xrange(n):
    plt.subplot(num_rows,num_cols,i+1)
    plt.imshow(p_template[i],interpolation='nearest',
               aspect=.1)

plt.subplots_adjust(wspace=0.00001)
fig.savefig('10_0_p_templates.png')
plt.show()

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('multipage_10_0_p_templates.pdf')

n=10
num_plots = n/2+1
num_cols = 2
cur_plot = 0

for i in xrange(num_plots):
    cur_sub_plot = 0
    while cur_sub_plot < num_cols and i*num_cols + cur_sub_plot < n:
        plt.subplot(1,num_cols,cur_sub_plot+1)
        plt.subplots_adjust(wspace=0.0001)
        plt.imshow(p_template[i*num_cols+cur_sub_plot],interpolation='nearest',
                   aspect=.1)
        cur_sub_plot += 1
    pp.savefig()






pp.close()    


# want to also check the divergence between the different templates
# do length normalization of the templates after the mixtures have been
# estimated and try that as a method for estimating these mixture components
# then try experiment with different number of mixture components

#
# the experiment indicates that 1 mixture component does the best in
# terms of likelihood, what we are really looking for is degenerate components
#
