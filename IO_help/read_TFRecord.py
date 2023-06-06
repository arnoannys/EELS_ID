import tensorflow as tf
import matplotlib.pyplot as plt
def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def parse_tfr_element(element):

    data = {
        'label_multilabel': tf.io.FixedLenFeature([], tf.string),
        'label_quantif': tf.io.FixedLenFeature([], tf.string),
        'raw_spec': tf.io.FixedLenFeature([], tf.string),
    }

    content = tf.io.parse_single_example(element, data)

    label_id = content['label_multilabel']
    label_quantif = content['label_quantif']
    raw_spec = content['raw_spec']

    feature = tf.io.parse_tensor(raw_spec, out_type=tf.float64)
    feature = tf.reshape(feature, shape=[3072])
    label_id = tf.io.parse_tensor(label_id, out_type=tf.float64)
    label_quantif = tf.io.parse_tensor(label_quantif, out_type=tf.float64)
    label = label_id  # or pick label_quantif or both as a dict
    return (feature, label)


def get_dataset(filename):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(
        parse_tfr_element
    )

    return dataset

dataset = get_dataset('Core-loss EELS TFRecord/trainingset/TRAIN_O.tfrecords').batch(4)


for data, label in dataset.take(10):
    #do something with first element in each batch
    plt.plot(data[0])
    print(label[0])

plt.show()
