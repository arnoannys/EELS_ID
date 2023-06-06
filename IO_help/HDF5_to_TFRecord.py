import tensorflow as tf
import h5py
import numpy as np
####################################################################################################
#HELPER FUNCTIONS
###################################################################################################
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
      value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def create_hdf5_generator(db_path):
    db = h5py.File(db_path)
    db_size = db['spectra'].shape[0]

    while True:
        for i in range(db_size):
            spectra = db['spectra'][i]
            label_indetif = db['labels_identification'][i]
            label_quant = db['labels_quantification'][i]

            yield spectra, label_indetif, label_quant


def write_spec_to_tfr(spec, label_multilabel, label_quantif, writer):
    #define the dictionary -- the structure -- of our single example
    data = {
        'raw_spec' : _bytes_feature(serialize_array(spec)),
        'label_multilabel' : _bytes_feature(serialize_array(label_multilabel)),
        'label_quantif' : _bytes_feature(serialize_array(label_quantif))
    }

    out = tf.train.Example(features=tf.train.Features(feature=data))
    writer.write(out.SerializeToString())



per_sys  = ["Be","B", "C", "N", "O", "F" , "Ne", "Na", "Mg", "Al", "Si", "P","S",  "Cl", "Ar"
               , "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se"
               , "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn"
               , "Sb" , "Te", "I", "Xe", "Cs", "Ba","La", "Ce", "Pr", "Nd","Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
               "Ho", "Er" , "Tm", "Yb","Lu", "Hf", "Ta", "W", "Re", "Os","Ir","Pt","Au","Hg","Tl","Pb","Bi"]

######################################################################################################################
for element in per_sys:
    filename=f"trainingset/TRAIN_{element}.tfrecords"
    db_path = f'trainingset/TRAIN_{element}.hdf5'
    hdf5_gen = create_hdf5_generator(db_path)
    with tf.io.TFRecordWriter(filename) as writer:
        curr = 0
        for sample, label_identif, label_quant in hdf5_gen:
            if curr >= 7000:
                break
            curr += 1
            sample = sample.astype(np.double)
            label_identif = label_identif.astype(np.double)
            label_quant = label_quant.astype(np.double)
            write_spec_to_tfr(sample,label_identif,label_quant,writer)

for element in per_sys:
    filename=f"testset/TEST_{element}.tfrecords"
    db_path = f'testset/TEST_{element}.hdf5'
    hdf5_gen = create_hdf5_generator(db_path)
    with tf.io.TFRecordWriter(filename) as writer:
        curr = 0
        for sample, label_identif, label_quant in hdf5_gen:
            if curr >= 1500:
                break
            curr += 1
            sample = sample.astype(np.double)
            label_identif = label_identif.astype(np.double)
            label_quant = label_quant.astype(np.double)
            write_spec_to_tfr(sample,label_identif,label_quant,writer)

for element in per_sys:
    filename=f"validationset/VALIDATION_{element}.tfrecords"
    db_path = f'validationset/VALIDATION_{element}.hdf5'
    hdf5_gen = create_hdf5_generator(db_path)
    with tf.io.TFRecordWriter(filename) as writer:
        curr = 0
        for sample, label_identif, label_quant in hdf5_gen:
            if curr >= 600:
                break
            curr += 1
            sample = sample.astype(np.double)
            label_identif = label_identif.astype(np.double)
            label_quant = label_quant.astype(np.double)
            write_spec_to_tfr(sample,label_identif,label_quant,writer)

for element in per_sys:
    filename=f"single_element_spectra/CONFMAT_{element}.tfrecords"
    db_path = f'single_element_spectra/CONFMAT_{element}.hdf5'
    hdf5_gen = create_hdf5_generator(db_path)
    with tf.io.TFRecordWriter(filename) as writer:
        curr = 0
        for sample, label_identif, label_quant in hdf5_gen:
            if curr >= 100:
                break
            curr += 1
            sample = sample.astype(np.double)
            label_identif = label_identif.astype(np.double)
            label_quant = label_quant.astype(np.double)
            write_spec_to_tfr(sample,label_identif,label_quant,writer)




