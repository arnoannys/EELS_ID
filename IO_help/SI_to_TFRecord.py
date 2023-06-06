import numpy as np
import tensorflow as tf
from scipy import interpolate
import hyperspy.api as hs
import matplotlib.pyplot as plt
import json
import argparse


parser = argparse.ArgumentParser()
####################################################################################################
#TFR FUNCTIONS
###################################################################################################
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

def write_spec_to_tfr(spec,idx_0,idx_1,sum,writer):

    data = {
        'raw_spec' : _bytes_feature(serialize_array(spec)),
        'idx_0' : _int64_feature(idx_0),
        'idx_1' : _int64_feature(idx_1),
        'sum' : _float_feature(sum)
    }

    out = tf.train.Example(features=tf.train.Features(feature=data))
    writer.write(out.SerializeToString())

####################################################################################################

def spline(energy_ax_old,energy_ax_new,spec):
    tck = interpolate.splrep(energy_ax_old, spec, s=0)
    new_spec = interpolate.splev(energy_ax_new, tck, der=0)
    return new_spec

#######################################################################################################
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--extension', type=str, required=True)
parser.add_argument('--offset', required=True)
args = parser.parse_args()
s = hs.load(f"SI_data/{args.name}.{args.extension}", signal_type="EELS")

x_step = s.axes_manager[0].scale
x_step_unit = s.axes_manager[0].units
x_range = s.axes_manager[0].size
y_step = s.axes_manager[1].scale
y_step_unit = s.axes_manager[1].units
y_range = s.axes_manager[1].size

E_range = s.axes_manager[2].size
Eaxis = s.axes_manager[2]
goal_disp = 1
goal_length = 3072
#########################################################################################################
Ebinfactor = goal_disp/Eaxis.scale
s = s.rebin((x_range,y_range,E_range/Ebinfactor))
s.axes_manager[2].offset += int(args.offset)
Eaxis = s.axes_manager[2]

energy_ax_old = np.arange(Eaxis.offset, Eaxis.offset+Eaxis.size, Eaxis.scale)
start = np.floor(energy_ax_old[0])
end =  np.ceil(energy_ax_old[-1])
energy_ax_mid = np.arange(start, end, goal_disp)
energy_ax_final = np.arange(0,goal_length,goal_disp)

filename=f"SI_data/{args.name}.tfrecords"
with tf.io.TFRecordWriter(filename) as writer: 
    for idx_x in range(x_range):
        for idx_y in range(y_range):
            
            spec = np.swapaxes(s.data,0,1)[idx_x,idx_y]
            sum = np.sum(spec)
            mid_spec = spline(energy_ax_old=energy_ax_old,energy_ax_new=energy_ax_mid,spec=spec)

            final_spec = np.zeros(len(energy_ax_final))
            final_spec[int(start):int(end)] = mid_spec
            final_spec[0:76] = 0
            final_spec = final_spec/np.max(final_spec)
            write_spec_to_tfr(final_spec,idx_x,idx_y,sum,writer)


metadata = {
    'idx_0_range' : x_range,
    'idx_1_range' : y_range,
    'idx_0_step' : x_step,
    'idx_1_step' : y_step,
    'idx_0_units' : x_step_unit,
    'idx_1_units' :  y_step_unit
}
json.dump( metadata, open( f"SI_data/{args.name}_metadata.json", 'w' ) )
with open(f"SI_data/{args.name}_metadata.json", 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

