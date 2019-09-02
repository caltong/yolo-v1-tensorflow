import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import yolo.config as cfg

"""
VOC2007处理
"""


class pascal_voc(object):
    """
    VOC2007数据集处理的类，主要用来获取训练集图片文件，以及生成对应的标签文件
    """

    def __init__(self, phase, rebuild=False):
        """
        准备训练或者测试的数据
        :param phase: 传入字符串 'train'：表示训练
                              'test'：测试
        :param rebuild: 否重新创建数据集的标签文件，保存在缓存文件夹下
        """
        self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')  # VOCdevkit文件夹路径
        self.data_path = os.path.join(self.devkil_path, 'VOC2007')  # VOC2007文件夹路径
        self.cache_path = cfg.CACHE_PATH  # catch文件所在路径
        self.batch_size = cfg.BATCH_SIZE  # 批大小
        self.image_size = cfg.IMAGE_SIZE  # 图像大小
        self.cell_size = cfg.CELL_SIZE  # 单元格大小
        self.classes = cfg.CLASSES  # VOC2007数据集类别名
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))  # 类别名->索引的dict
        self.flipped = cfg.FLIPPED  # 是否采用水平镜像扩充数据集
        self.phase = phase  # 训练或测试
        self.rebuild = rebuild  # 是否重新创建数据集标签文件
        self.cursor = 0  # 从gt_labels加载数据，cursor表明当前读取到第几个
        self.epoch = 1  # 存放当前训练的轮数

        # 存放数据集的标签 是一个list 每一个元素都是一个dict，对应一个图片
        # 如果我们在配置文件中指定flipped=True，则数据集会扩充一倍，每一张原始图片都有一个水平对称的镜像文件
        #      imname：图片路径
        #      label：图片标签
        #      flipped：图片水平镜像？
        self.gt_labels = None
        self.prepare()  # 加载数据集标签  初始化gt_labels

    def get(self):
        """
        加载数据集 每次读取batch大小的图片以及图片对应的标签

        :return:
            images:读取到的图片数据 [45,448,448,3]
            labels:对应的图片标签 [45,7,7,25]
        """
        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3))  # [45,448,448,3]
        labels = np.zeros(
            (self.batch_size, self.cell_size, self.cell_size, 25))  # [45,7,7,25]
        count = 0

        # 一次加载batch_size个图片数据
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']  # 获取图片路径
            flipped = self.gt_labels[self.cursor]['flipped']  # 是否使用水平镜像？
            images[count, :, :, :] = self.image_read(imname, flipped)  # 读取图片数据
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']  # 读取图片标签
            count += 1
            self.cursor += 1
            # 如果读取完一轮数据，则当前cursor置为0，当前训练轮数+1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)  # 打乱数据集
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def image_read(self, imname, flipped=False):
        """
        读取图片

        :param imname: 图片路径
        :param flipped: 图片是否水平镜像处理？
        :return: image：图片数据 [448,448,3]
        """
        image = cv2.imread(imname)  # 读取图片数据
        image = cv2.resize(image, (self.image_size, self.image_size))  # 缩放处理
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)  # BGR->RGB  uint->float32
        image = (image / 255.0) * 2.0 - 1.0  # 归一化处理 [-1.0,1.0]

        # 宽倒序  即水平镜像
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self):
        """
        初始化数据集的标签，保存在变量gt_labels中
        :return: gt_labels:返回数据集的标签 是一个list  每一个元素对应一张图片，是一个dict
                 imname：图片文件路径
                 label：图片文件对应的标签 [7,7,25]的矩阵
                 flipped：是否使用水平镜像？ 设置为False
        """
        gt_labels = self.load_labels()  # 加载数据集的标签
        # 如果水平镜像，则追加一倍的训练数据集
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            # 深度拷贝
            gt_labels_cp = copy.deepcopy(gt_labels)
            # 遍历每一个图片标签
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True  # 设置flipped属性为True
                gt_labels_cp[idx]['label'] = \
                    gt_labels_cp[idx]['label'][:, ::-1, :]  # 目标所在格子也进行水平镜像 [7,7,25]
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = \
                                self.image_size - 1 - \
                                gt_labels_cp[idx]['label'][i, j, 1]  # 置信度==1，表示这个格子有目标
            gt_labels += gt_labels_cp  # 追加数据集的标签   后面的是由原数据集标签扩充的水平镜像数据集标签
        np.random.shuffle(gt_labels)  # 打乱数据集的标签
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self):
        """
        加载数据集标签

        :return:
            gt_labels：是一个list  每一个元素对应一张图片，是一个dict
            imname：图片文件路径
            label：图片文件对应的标签 [7,7,25]的矩阵
            flipped：是否使用水平镜像？ 设置为False
        """
        cache_file = os.path.join(
            self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl')  # 缓冲文件名：即用来保存数据集标签的文件

        # 文件存在，且不重新创建则直接读取
        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)

        # 如果缓冲文件目录不存在，创建
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        # 获取训练测试集的数据文件名
        if self.phase == 'train':
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        # 获取测试集的数据文件名
        else:
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'test.txt')
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]

        # 存放图片的标签，图片路径，是否使用水平镜像？
        gt_labels = []
        # 遍历每一张图片的信息
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index)  # 读取每一张图片的标签label [7,7,25]
            if num == 0:
                continue
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')  # 图片文件路径
            gt_labels.append({'imname': imname,
                              'label': label,
                              'flipped': False})  # 保存该图片的信息
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)  # 保存
        return gt_labels

    def load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.

        args:
            index：图片文件的index

        return :
            label：标签 [7,7,25]
                      0:1：置信度，表示这个地方是否有目标
                      1:5：目标边界框  目标中心，宽度和高度(这里是实际值，没有归一化)
                      5:25：目标的类别
            len(objs)：objs对象长度
        """

        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')  # 获取图片文件名路径
        im = cv2.imread(imname)  # 读取数据
        h_ratio = 1.0 * self.image_size / im.shape[0]  # 宽和高缩放比例
        w_ratio = 1.0 * self.image_size / im.shape[1]
        # im = cv2.resize(im, [self.image_size, self.image_size])
        # 用于保存图片文件的标签
        label = np.zeros((self.cell_size, self.cell_size, 25))
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')  # 图片文件的标注xml文件
        tree = ET.parse(filename)
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based 当图片缩放到image_size时，边界框也进行同比例缩放
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            # 根据图片的分类名 ->类别index 转换
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
            # 计算边框中心点x,y,w,h(没有归一化)
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            # 计算当前物体的中心在哪个格子中
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            # 表明该图片已经初始化过了
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1  # 置信度，表示这个地方有物体
            label[y_ind, x_ind, 1:5] = boxes  # 物体边界框
            label[y_ind, x_ind, 5 + cls_ind] = 1  # 物体的类别

        return label, len(objs)
