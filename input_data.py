import numpy as np
import data

class DataSet(object):
    def __init__(self, images, labels, generative_labels):
        assert images.shape[0] == labels.shape[0], (
            "images.shape: %s labels.shape: %s" % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._dim1 = images.shape[1]
        self._dim2 = images.shape[2]
        self._num_labels = labels.shape[1]
        try:
            self._num_channels = images.shape[3]
        except:
            self._num_channels = 1

        self._images = images
        self._labels = labels
        self._gen_labels = generative_labels

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def gen_labels(self):
        return self._gen_labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def dim1(self):
        return self._dim1

    @property
    def dim2(self):
        return self._dim2

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def num_labels(self):
        return self._num_labels


def read_data_sets(which_dataset, channel):
    class DataSets(object):
        pass
    data_sets = DataSets()

    if which_dataset.lower() == 'mnist':
        train_images_path = data.may_download_file('train_data_mnist')
        test_images_path = data.may_download_file('test_data_mnist')
        train_labels_path = data.may_download_file('train_labels_mnist')
        test_labels_path = data.may_download_file('test_labels_mnist')
        train_images = np.load(train_images_path)
        train_labels = np.load(train_labels_path)
        test_images = np.load(test_images_path)
        test_labels = np.load(test_labels_path)
        train_images = train_images.reshape([train_images.shape[0],
                                             train_images.shape[1],
                                             train_images.shape[2], 1])
        test_images = test_images.reshape([test_images.shape[0],
                                           test_images.shape[1],
                                           test_images.shape[2], 1])

        generative_labels = []

    elif which_dataset.lower() == 'cells_binary_classifier':
        train_images_path = data.may_download_file('binary_train_images')
        test_images_path = data.may_download_file('binary_test_images')
        train_labels_path = data.may_download_file('binary_train_labels')
        test_labels_path = data.may_download_file('binary_test_labels')
        train_images = np.load(train_images_path)
        train_labels = np.load(train_labels_path)
        test_images = np.load(test_images_path)
        test_labels = np.load(test_labels_path)

        generative_labels = []

        if channel.lower() == 'first':
            train_images = train_images[:, :, :, 0]
            test_images = test_images[:, :, :, 0]
            train_images = train_images.reshape([train_images.shape[0],
                                                 train_images.shape[1],
                                                 train_images.shape[2], 1])
            test_images = test_images.reshape([test_images.shape[0],
                                               test_images.shape[1],
                                               test_images.shape[2], 1])
        elif channel.lower() == 'second':
            train_images = train_images[:, :, :, 1]
            test_images = test_images[:, :, :, 1]
            train_images = train_images.reshape([train_images.shape[0],
                                                 train_images.shape[1],
                                                 train_images.shape[2], 1])
            test_images = test_images.reshape([test_images.shape[0],
                                               test_images.shape[1],
                                               test_images.shape[2], 1])
        elif channel.lower() == 'both':
            pass
        else:
            raise ValueError('please state the correct channel to train on: first, second or both')

    elif which_dataset.lower() == 'unlabelled_cells_filtered':
        train_images_path = data.may_download_file('72k_dataset_filtered_normalized_imgages')
        train_images = np.load(train_images_path)
        # train_labels_path = data.may_download_file('72k_dataset_filtered_normalized_labels')
        # train_labels = np.load(train_labels_path)
        train_labels = np.empty(shape=(train_images.shape[0], 2))
        test_images = np.empty(shape=train_images.shape)
        test_labels = np.empty(shape=train_labels.shape)

        generative_label_path = data.may_download_file('72k_dataset_filtered_normalized_labels')
        generative_labels = np.load(generative_label_path)

        if channel.lower() == 'first':
            train_images = train_images[:, :, :, 0]
            test_images = test_images[:, :, :, 0]
            train_images = train_images.reshape([train_images.shape[0],
                                                 train_images.shape[1],
                                                 train_images.shape[2], 1])
            test_images = test_images.reshape([test_images.shape[0],
                                               test_images.shape[1],
                                               test_images.shape[2], 1])
        elif channel.lower() == 'second':
            train_images = train_images[:, :, :, 1]
            test_images = test_images[:, :, :, 1]
            train_images = train_images.reshape([train_images.shape[0],
                                                 train_images.shape[1],
                                                 train_images.shape[2], 1])
            test_images = test_images.reshape([test_images.shape[0],
                                               test_images.shape[1],
                                               test_images.shape[2], 1])
        elif channel.lower() == 'both':
            pass
        else:
            raise ValueError('please state the correct channel to train on: first, second or both')

    elif which_dataset.lower() == 'unlabelled_cells_filtered_newdata':
        train_images_path = data.may_download_file('72k_dataset_filtered_normalized_imgages_newdata')
        train_images = np.load(train_images_path).astype('float32')

        generative_label_path = data.may_download_file('72k_dataset_filtered_normalized_labels_newdata')
        generative_labels = np.load(generative_label_path)
        np.random.seed(1)
        #idx = np.random.choice(133083, 133083, replace=False)
        idx = np.random.choice(133083, 40000, replace=False)
        train_images = train_images[idx]
        generative_labels = generative_labels[idx]
        
        train_labels = np.empty(shape=(train_images.shape[0], 2))
        test_images = np.empty(shape=train_images.shape).astype('float32')
        test_labels = np.empty(shape=train_labels.shape)

        if channel.lower() == 'first':
            train_images = train_images[:, :, :, 0]
            test_images = test_images[:, :, :, 0]
            train_images = train_images.reshape([train_images.shape[0],
                                                 train_images.shape[1],
                                                 train_images.shape[2], 1])
            test_images = test_images.reshape([test_images.shape[0],
                                               test_images.shape[1],
                                               test_images.shape[2], 1])
        elif channel.lower() == 'second':
            train_images = train_images[:, :, :, 1]
            test_images = test_images[:, :, :, 1]
            train_images = train_images.reshape([train_images.shape[0],
                                                 train_images.shape[1],
                                                 train_images.shape[2], 1])
            test_images = test_images.reshape([test_images.shape[0],
                                               test_images.shape[1],
                                               test_images.shape[2], 1])
        elif channel.lower() == 'both':
            pass
        else:
            raise ValueError('please state the correct channel to train on: first, second or both')

    else:
        raise ValueError('Please choose between "cells" or "mnist" datasets')

    data_sets.train = DataSet(train_images, train_labels, generative_labels)
    data_sets.test = DataSet(test_images, test_labels, generative_labels)

    return data_sets
