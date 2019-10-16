import os.path
import config
from tools import create_alexnet
from preprocessing_RCNN import load_data
import tflearn


def train(network, X, Y, save_model_path=config.SAVE_MODEL_PATH):
    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='output')
    if os.path.isfile(save_model_path + '.index'):
        model.load(save_model_path)
        print('load model...')
    for _ in range(5):
        model.fit(X, Y, n_epoch=1, validation_set=0.1, shuffle=True,
                  show_metric=True, batch_size=64, snapshot_step=200,
                  snapshot_epoch=False, run_id='alexnet_oxflowers17')  # epoch = 1000
    # Save the model
    model.save(save_model_path)
    print('save model...')


def main():
    first_time = False
    x, y = load_data(list=config.TRAIN_LIST, cls=config.TRAIN_CLASS, save=first_time)
    net = create_alexnet()
    train(net, x, y)


if __name__ == '__main__':
    main()
