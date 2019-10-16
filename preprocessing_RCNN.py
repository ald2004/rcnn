import numpy as np
import pickle, gzip
import preprocessing_RCNN as prep
import cv2
import config
import codecs
import selectivesearch as ss
import tools, os


def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=cv2.INTER_CUBIC):
    img = cv2.resize(in_image, (new_width, new_height), resize_mode)
    if out_image:
        cv2.imwrite(out_image, img)
    return img


def load_data(list=config.TRAIN_LIST, cls=config.TRAIN_CLASS, save=False, save_path='dataset.pkl.zip'):
    print(f'list is:{list}, cls is :{cls}')
    if save:
        with codecs.open(list, 'r', 'utf-8') as f:
            train_list = f.readlines()
            labels, imgs = [], []
            for line in train_list:
                tmp = line.strip().split(' ')
                img = prep.resize_image(cv2.imread(tmp[0]), config.IMAGE_SIZE, config.IMAGE_SIZE)
                imgs.append(np.asarray(img, dtype='float32'))
                labels.append(int(tmp[1]))
            targets = np.array(labels).reshape(-1)
            labels = np.eye(cls)[targets].reshape(-1, cls)
        with gzip.open(save_path, 'wb') as listz:
            pickle.dump((imgs, labels), listz)
    else:
        with gzip.open(save_path, 'rb') as f:
            return pickle.load(f)
    return imgs, labels


def load_train_proposals(datafile, nbr_classes, save_path, threshold=0.5, is_svm=False, save=False):
    with open(datafile, 'r') as f:
        train_list = f.readlines()
        for i, line in enumerate(train_list):
            labels, imgs = [], []
            tmp = line.strip().split(' ')
            img = cv2.imread(tmp[0])
            _, regions = ss.selective_search(img, scale=500, sigma=0.9, min_size=10)
            candicates = set()
            for reg in regions:
                if (reg['size'] < 200) or (reg['rect'] in candicates) or (
                        reg['rect'][2] * reg['rect'][3] < 500): continue
                proposal_img, proposal_vertice = clip_pic(img, reg['rect'])
                if not len(proposal_img): continue
                [a, b, c] = np.shape(proposal_img)
                if not a or not b or not c: continue
                resized_proposal_img = resize_image(proposal_img, config.IMAGE_SIZE, config.IMAGE_SIZE)
                candicates.add(reg['rect'])
                imgs.append(resized_proposal_img.astype(np.float))
                # IOU
                ref_rect_int = [int(i) for i in tmp[2].split(',')]
                iou_val = IOU(ref_rect_int, proposal_vertice)
                index = int(tmp[1])
                if is_svm:
                    if iou_val < threshold:
                        labels.append(0)
                    else:
                        labels.append(index)
                else:
                    label = np.zeros(nbr_classes + 1)
                    if iou_val < threshold:
                        label[0] = 1
                    else:
                        label[index] = 1
                    labels.append(label)
            tools.view_bar("processing image of %s" % datafile.split('\\')[-1].strip(), i + 1, len(train_list))
            if save:
                np.save((os.path.join(save_path, tmp[0].split('/')[-1].split('.')[0].strip()) + '_data.npy'),
                        [imgs, labels])


def clip_pic(img, rect):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    x_1 = x + w
    y_1 = y + h
    # return img[x:x_1, y:y_1, :], [x, y, x_1, y_1, w, h]
    return img[y:y_1, x:x_1, :], [x, y, x_1, y_1, w, h]


# IOU Part 1
def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return if_intersect
    if if_intersect:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter


# IOU Part 2
def IOU(ver1, vertice2):
    # vertices in four points
    vertice1 = [ver1[0], ver1[1], ver1[0] + ver1[2], ver1[1] + ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3], vertice2[0], vertice2[2],
                                 vertice2[1], vertice2[3])
    if area_inter:
        area_1 = ver1[2] * ver1[3]
        area_2 = vertice2[4] * vertice2[5]
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    return False


def load_from_npy(data_path):
    images, labels = [], []
    data_list = os.listdir(data_path)
    # random.shuffle(data_list)
    for ind, d in enumerate(data_list):
        i, l = np.load(os.path.join(data_path, d))
        images.extend(i)
        labels.extend(l)
        tools.view_bar("load data of %s" % d, ind + 1, len(data_list))
    print(' ')
    return images, labels
