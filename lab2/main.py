# your code here
import os
import random
import sys
from functools import reduce

import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import numpy as np

import cifar10
from activations import softmax
from softmax import SoftmaxClassifier


def plots():
    x = np.asarray([20, 30, -15, 45, 39, -10])
    n = len(x)
    temperatures = [0.25, 0.75, 1, 1.5, 2, 5, 10, 20, 30]
    for temperature in temperatures:
        plt.plot(range(n), softmax(x, temperature))
    plt.legend(tuple(f"{temperature}" for temperature in temperatures))
    plt.show()
    array = np.asarray([20, 30, -15, 45, 39, -10])
    temperatures = [0.25, 0.75, 1, 1.5, 2, 5, 10, 20, 30]

    for idx in range(0, len(temperatures)):
        plot.subplot(3, 3, idx + 1)
        plot.bar(range(array.shape[0]), softmax(array, temperatures[idx]))
        plot.ylim([0, 1])
        plot.title(temperatures[idx])
    plot.show()


def cifar_labels():
    cifar_root_dir = 'cifar-10-batches-py'
    _, _, x_test, y_test = cifar10.load_ciaf10(cifar_root_dir)
    indices = np.random.choice(len(x_test), 15)

    display_images, display_labels = x_test[indices], y_test[indices]
    for idx, (image, label) in enumerate(zip(display_images, display_labels)):
        plot.subplot(3, 5, idx + 1)
        plot.imshow(image)
        plot.title(cifar10.LABELS[label])
        plot.tight_layout()
    plot.show()


def train():
    cifar_root_dir = 'cifar-10-batches-py'

    # the number of trains performed with different hyper-parameters
    search_iter = 10
    # the batch size
    batch_size = 200
    # number of training steps per training process
    train_steps = 5000

    # load cifar10 dataset
    x_train, y_train, x_test, y_test = cifar10.load_ciaf10(cifar_root_dir)

    # convert the training and test data to floating point
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # Reshape the training data such that we have one image per row
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))

    # pre-processing: subtract mean image
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_test -= mean_image

    # Bias trick - add 1 to each training example
    x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
    x_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))])

    # the search limits for the learning rate and regularization strength
    # we'll use log scale for the search
    lr_bounds = (-7, -2)
    reg_strength_bounds = (-4, -2)

    if not os.path.exists('train'):
        os.mkdir('train')

    best_acc = -1
    best_cls_path = ''

    learning_rates = [-7, -5]
    regularization_strengths = [3000, 80000]

    input_size_flattened = reduce((lambda a, b: a * b), x_train[0].shape)
    results = []

    for _ in range(0, search_iter):
        # use log scale for sampling the learning rate
        lr = pow(10, random.uniform(learning_rates[0], learning_rates[1]))
        reg_strength = random.uniform(
            regularization_strengths[0], regularization_strengths[1])

        cls = SoftmaxClassifier(
            input_shape=input_size_flattened, num_classes=cifar10.NUM_CLASSES)
        history = cls.fit(x_train, y_train, lr=lr, reg_strength=reg_strength,
                          steps=train_steps, bs=batch_size)

        results.append({
            'lr': lr,
            'reg': reg_strength,
            'history': history
        })

        y_train_pred = cls.predict(x_train)
        y_val_pred = cls.predict(x_test)

        train_acc = np.mean(y_train == y_train_pred)

        test_acc = np.mean(y_test == y_val_pred)
        sys.stdout.write(
            '\rlr {:.4f}, reg_strength{:.2f}, test_acc {:.2f}; train_acc {:.2f}'.format(lr, reg_strength, test_acc,
                                                                                        train_acc))
        cls_path = os.path.join(
            'train', 'softmax_lr{:.4f}_reg{:.4f}-test{:.2f}.npy'.format(lr, reg_strength, test_acc))
        cls.save(cls_path)

        if test_acc > best_acc:
            best_acc = test_acc
            best_cls_path = cls_path

    num_rows = search_iter // 5 + 1
    for idx, res in enumerate(results):
        plt.subplot(num_rows, 5, idx + 1)
        plt.plot(res['history'])
    plt.show()

    best_softmax = SoftmaxClassifier(
        input_shape=input_size_flattened, num_classes=cifar10.NUM_CLASSES)
    best_softmax.load(best_cls_path)

    plt.rcParams['image.cmap'] = 'gray'
    # now let's display the weights for the best model
    weights = best_softmax.get_weights((32, 32, 3))
    w_min = np.amin(weights)
    w_max = np.amax(weights)

    for idx in range(0, cifar10.NUM_CLASSES):
        plt.subplot(2, 5, idx + 1)
        # normalize the weights
        template = 255.0 * \
            (weights[idx, :, :, :].squeeze() - w_min) / (w_max - w_min)
        template = template.astype(np.uint8)
        plt.imshow(template)
        plt.title(cifar10.LABELS[idx])

    plt.show()


plots()
cifar_labels()
# train()
