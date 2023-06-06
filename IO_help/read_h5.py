import h5py
import numpy as np 
import matplotlib.pyplot as plt

def create_hdf5_generator(db_path, batch_size):
    db = h5py.File(db_path)
    db_size = db['spectra'].shape[0]

    while True:
        for i in np.arange(0, db_size, batch_size):
            spectra = db['spectra'][i:i+batch_size]
            labels_id = db['labels_identification'][i:i+batch_size]
            labels_quant = db['labels_quantification'][i:i+batch_size]

            yield spectra, labels_id, labels_quant

def hdf5_read(db_path):
    batch_size = 4
    hdf5_gen = create_hdf5_generator(db_path, batch_size)
    total_nr_spec = 40 # number of spectra to get. note that this will loop over the same spectra if you exceed the available number of spectra
    max_iter = total_nr_spec/batch_size

    curr = 0
    for spectra, labels_id, labels_quant in hdf5_gen:
        if curr >= max_iter:
            break
        curr += 1

        #do something with first element in each batch
        plt.plot((spectra[0]))
        print(labels_id[0])
        print(labels_quant[0])


hdf5_read(db_path='Core-loss EELS HDF5/trainingset/TRAIN_O.hdf5')
plt.show()