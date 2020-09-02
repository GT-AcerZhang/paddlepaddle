
import numpy as np
import gzip
import json
import random

IMG_ROWS, IMG_COLS = 28, 28
data_path = './mnist.json.gz'


def load_data(mode='train', batch_size=100):
    print('loading mnist dataset from {} ......'.format(data_path))
    data = json.load(gzip.open(data_path))
    train_set, val_set, eval_set = data

    # 根据输入mode参数决定使用训练集，验证集还是测试
    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'eval':
        imgs = eval_set[0]
        labels = eval_set[1]
    index_list = list(range(len(imgs)))

    # 返回对象
    def data_generator():
        if mode == 'train':
            random.shuffle(index_list)
        img_lst, label_lst = [], []
        for i in index_list:
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
            img_lst.append(img)
            label_lst.append(label)
            if len(img_lst) == batch_size:
                yield np.array(img_lst), np.array(label_lst)
                # batch已经形成，清空数据缓存，读取下一个batch
                img_lst, label_lst = [], []
        # 如果剩余数据的数目小于BATCHSIZE，返回mini-batch
        if len(img_lst) > 0:
            yield np.array(img_lst), np.array(label_lst)
    return data_generator
    
