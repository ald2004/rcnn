import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
import config
import os


def show_rect(img_path, regions):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    img = skimage.io.imread(img_path)
    ax.imshow(img)
    for x, y, w, h in regions:
        rect = patches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()


def show_rect_dict(img_path, regions):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    if isinstance(img_path, str):
        img_path = skimage.io.imread(img_path)
    ax.imshow(img_path.astype(np.uint8))
    for region in regions:
        x, y, w, h = region['rect']
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1))
    plt.show()


def show_img(img):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    if isinstance(img, str):
        img = skimage.io.imread(img)
    img = img.astype(np.uint8)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def view_bar(message, num, total):
    import math
    import sys
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


def create_alexnet(nbr_class=config.TRAIN_CLASS, restore=False):
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.normalization import local_response_normalization
    from tflearn.layers.estimator import regression
    network = input_data(shape=[None, config.IMAGE_SIZE, config.IMAGE_SIZE, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, nbr_class, activation='softmax', restore=restore)
    network = regression(network, optimizer='momentum', loss='categorical_crossentropy', learning_rate=0.001)
    return network

#config.SAVE_MODEL_PATH, config.FINE_TUNE_MODEL_PATH
def fine_tune_Alexnet(network, X, Y, save_model_path, fine_tune_model_path):
    import tflearn
    model = tflearn.DNN(network, checkpoint_path='rcnn_model_alexnet', max_checkpoints=2, tensorboard_dir='output_RCNN',
                        tensorboard_verbose=2)
    if os.path.isfile(fine_tune_model_path + '.index'):
        print(f'Loading fine tuned model ...')
        model.load(fine_tune_model_path)
    elif os.path.isfile(save_model_path+ '.index'):
        print("Loading the alexnet ...")
        model.load(save_model_path)
    else:
        print("No file to load, error")
        return False
    model.fit(X, Y, n_epoch=1, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='alexnet_rcnnflowers2')
    # Save the model
    model.save(fine_tune_model_path)
