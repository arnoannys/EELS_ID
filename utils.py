import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap , Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
from matplotlib.collections import LineCollection
from data_parameters import data_param
####################################################################################################""
#TFRECORD
###################################################################################################""
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
      value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def parse_tfr_element(element):
    data = {
        'label_multilabel':tf.io.FixedLenFeature([], tf.string),
        'label_quantif':tf.io.FixedLenFeature([], tf.string),
        'raw_spec' : tf.io.FixedLenFeature([], tf.string),
           }

      
    content = tf.io.parse_single_example(element, data)
    
    label_multilabel = content['label_multilabel']
    raw_spec = content['raw_spec']
  
    feature = tf.io.parse_tensor(raw_spec, out_type=tf.float64)
    feature = tf.reshape(feature, shape=[3072])
    label_multilabel = tf.io.parse_tensor(label_multilabel, out_type=tf.float64)
    return (feature, label_multilabel)

def get_dataset(filename):
    #create the dataset
    dataset = tf.data.TFRecordDataset(filename)

    #pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )
      
    return dataset


######################################################################################################
# CUSTOM LOSS FUNCTION
######################################################################################################
@tf.function
def custom_loss(y, y_hat):
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 
    macro_cost = tf.reduce_mean(cost) 
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y,y_hat)
    return macro_cost + BCE

######################################################################################################

class Visuals():
    def __init__(self):
        self.per_sys = data_param['per_sys']
        self.N_elem = len(self.per_sys)
      
      
    ##########################################################################################################
    def string_and_order_output(self,output):
        
        output = np.around(output, decimals=2)
        order = np.argsort(output)
        n_nonzero = np.count_nonzero(output)
        n_zero = self.N_elem-n_nonzero
        ps_order = []
        values = []
        more = None
        if n_nonzero <= 5:
            for i in range(n_zero,self.N_elem):
                ps_order = np.append(ps_order, self.per_sys[order[i]])
                values = np.append(values, output[order[i]])
                more = False
        if n_nonzero > 5:
            for i in range(self.N_elem-5,self.N_elem):
                ps_order = np.append(ps_order, self.per_sys[order[i]])
                values = np.append(values, output[order[i]])
                more = True

                
        ps_order = np.flip(ps_order)
        values = np.flip(values)

        string = ''
        for i in range(len(values)):
            string = string + str(i+1) + ":  " + ps_order[i] + "    " +  "{:.2f}".format(values[i]) + '\n'
        if more:
            string = string + "...."
        return string

    def output_to_name(self,label,threshold) -> str:
        name = ''
        for i in range(self.N_elem):
            if label[i] >= threshold:
                name = name + ' ' + self.per_sys[i]
        return name



    def visual_prediction(self,model, dataset,start = 0, end = 10, threshold =0.5 , save=False,filename=''):
    

        text_kwargs_1 = dict(ha='center', va='bottom', fontsize=10, weight='bold')
        text_kwargs_2 = dict(ha='center', va='center', fontsize=10, weight='bold')

        #custom colormap
        cmap= cm.get_cmap('Greens', 256)
        newcolors = cmap(np.linspace(0.1, 1, 256))
        newcmap = ListedColormap(newcolors)


        #shape the periodic system
        GRID = np.zeros([10, 18])
        GRID[0, 1:17] = np.nan
        GRID[1:3, 2:12] = np.nan
        GRID[7, :] = np.nan
        GRID[5:7,2] = np.nan
        GRID[8:10, 0:3] = np.nan
        

        #map consecutive indices of list to positions
        mapping = [[1,1],[1, 12], [1, 13]]
        for i in range(14, 18):
            mapping = np.vstack((mapping, [1, i]))
        mapping = np.vstack((mapping, [2, 0]))
        mapping = np.vstack((mapping, [2, 1]))
        for i in range(12, 18):
            mapping = np.vstack((mapping, [2, i]))
            
        for j in range(3, 5):
            for i in range(18):
                mapping = np.vstack((mapping, [j, i]))
                
        mapping = np.vstack((mapping, [5, 0]))
        mapping = np.vstack((mapping, [5, 1]))
        for i in range(3, 18):  
            mapping = np.vstack((mapping, [8, i]))
        for i in range(3, 15):
            mapping = np.vstack((mapping, [5, i]))


        N = end-start

        #predict
        for spectra, labels in dataset:
            output = model.predict(spectra)

        #prepare full output image
        fig, ax = plt.subplots(N,2, figsize=(20,N*5), num = start)

        #loop over each spec
        for idx in range(start,end):
            out = output[idx]
            grid = GRID
            #map the output
            idx_ax = idx%N
            ax[idx_ax,0].plot(spectra[idx], 'k')
            ax[idx_ax,0].set_yticklabels([])
            ax[idx_ax,0].set_yticks([])
            ax[idx_ax,0].set_xticks([0,3071])
            ax[idx_ax,0].set_xticklabels(['0 eV' , '3071 eV'])
            ax[idx_ax,0].tick_params(direction='in')
            ax[idx_ax,0].set_title(f' True label: {self.output_to_name(labels[idx],threshold)}')
            for j in range(len(out)):
                grid[mapping[j][0], mapping[j][1]] = out[j]
            
              #plot
                if grid[mapping[j][0], mapping[j][1]] != np.nan:
                    name = self.per_sys[j]
                    ax[idx_ax,1].text(mapping[j][1],mapping[j][0],  f'{name}',text_kwargs_1)
                    ax[idx_ax,1].text(mapping[j][1],mapping[j][0],  f'\n{grid[mapping[j][0], mapping[j][1]]:.2f}' , ha='center', va='center', fontsize=8)
            im = ax[idx_ax,1].imshow(grid, cmap = newcmap, vmin = 0, vmax = 1)
            ax[idx_ax,1].text(6, 1, self.string_and_order_output(out) , **text_kwargs_2)
            ax[idx_ax,1].set_xticks(np.arange(-0.5, 18, 1))
            ax[idx_ax,1].set_yticks(np.arange(-0.5, 10, 1))
            ax[idx_ax,1].set_xticklabels([])
            ax[idx_ax,1].set_yticklabels([])
            ax[idx_ax,1].grid(color='w', linestyle='-', linewidth=2)
            ax[idx_ax,1].set_title(f'Predicted label: {self.output_to_name(out,threshold)}')
            ax[idx_ax,1].tick_params(length = 0)
        plt.tight_layout()  
        if save == True:
            plt.savefig(filename)
      
