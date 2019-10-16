import os
import config
import preprocessing_RCNN as pres
from tools import create_alexnet
from tools import fine_tune_Alexnet


def main():
    data_set = config.FINE_TUNE_DATA
    if not len(os.listdir(data_set)):
        print(f'generating  data...')
        pres.load_train_proposals(config.FINE_TUNE_LIST, 2, save=True, save_path=data_set)
    print(f'Loading data...')
    x, y = pres.load_from_npy(data_set)
    restore = False
    if os.path.isfile(config.FINE_TUNE_MODEL_PATH + '.index'):
        restore = True
        print(f'Continue fine-tune ...')
    anet = create_alexnet(nbr_class=config.FINE_TUNE_CLASS, restore=restore)
    fine_tune_Alexnet(anet, x, y, config.SAVE_MODEL_PATH, config.FINE_TUNE_MODEL_PATH)


if __name__ == '__main__':
    main()
