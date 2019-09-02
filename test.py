import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer


class Detector(object):

    def __init__(self, net, weight_file):
        """
        构造函数
        利用 cfg 文件对网络参数进行初始化，
        其中 offset 的作用应该是一个定长的偏移
        boundery1和boundery2 作用是在输出中确定每种信息的长度（如类别，置信度等）。
        其中 boundery1 指的是对于所有的 cell 的类别的预测的张量维度，所以是 self.cell_size * self.cell_size * self.num_class
        boundery2 指的是在类别之后每个cell 所对应的 bounding boxes 的数量的总和，所以是self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        :param net: YOLONet对象
        :param weight_file: 检查点文件路径
        """
        self.net = net  # yolo网络
        self.weights_file = weight_file  # 检查点文件路径

        self.classes = cfg.CLASSES  # VOC2007数据类别名
        self.num_class = len(self.classes)  # 数据类别数
        self.image_size = cfg.IMAGE_SIZE  # 图像尺寸
        self.cell_size = cfg.CELL_SIZE  # grid cell 尺寸
        self.boxes_per_cell = cfg.BOXES_PER_CELL  # 单个cell中box数量
        self.threshold = cfg.THRESHOLD  # 阈值参数
        self.iou_threshold = cfg.IOU_THRESHOLD  # IOU阈值

        # 将网络输出分离为类别和置信度以及边界框的大小，输出维度为7*7*20 + 7*7*2 + 7*7*2*4=1470
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + \
                         self.cell_size * self.cell_size * self.boxes_per_cell

        # 运行图之前，初始化变量
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # 恢复模型
        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def draw_result(self, img, result):
        """
        在原图上绘制边界框，以及附加信息

        :param img: 原始图片数据
        :param result: yolo网络目标检测到的边界框，list类型 每一个元素对应一个目标框
                  包含{类别名,x_center,y_center,w,h,置信度}
        """
        # 遍历所有边界框
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)  # 绘制矩形框(目标边界框) 矩形左上角，矩形右下角
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)  # 绘制矩形框，用于存放类别名称，使用灰度填充
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA  # 线型
            cv2.putText(
                img, result[i][0] + ' : %.2f' % result[i][5],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType)  # 绘制文本信息 写上类别名和置信度

    def detect(self, img):
        """
         图片目标检测

        :param img: 原始图片数据
        :return: 返回检测到的边界框，list类型 每一个元素对应一个目标框
            包含{类别名,x_center,y_center,w,h,置信度}
        """
        img_h, img_w, _ = img.shape  # 获取图片的高和宽
        inputs = cv2.resize(img, (self.image_size, self.image_size))  # 图片缩放 [448,448,3]
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)  # BGR->RGB  uint->float32
        inputs = (inputs / 255.0) * 2.0 - 1.0  # 归一化处理 [-1.0,1.0]
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))  # reshape [1,448,448,3]

        result = self.detect_from_cvmat(inputs)[0]  # 获取网络输出第一项(即第一张图片) [1,1470]

        # 对检测的图片的边界框进行缩放处理，一张图片可以有多个边界框
        for i in range(len(result)):
            # x_center, y_center, w, h都是真实值，分别表示预测边界框的中心坐标，宽和高，都是浮点型
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)

        return result

    def detect_from_cvmat(self, inputs):
        # 返回网络最后一层，激活函数处理之前的值  形状[None,1470]
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        results = []
        # 对网络输出每一行数据进行处理
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

        # 返回处理后的结果
        return results

    def interpret_output(self, output):
        """
        对yolo网络输出进行处理

        :param output: yolo网络输出的每一行数据 大小为[1470,]
                    0：7*7*20：表示预测类别
                    7*7*20:7*7*20 + 7*7*2:表示预测置信度，即预测的边界框与实际边界框之间的IOU
                    7*7*20 + 7*7*2：1470：预测边界框    目标中心是相对于当前格子的，宽度和高度的开根号是相对当前整张图像的(归一化的)
        :return: yolo网络目标检测到的边界框，list类型 每一个元素对应一个目标框
                  包含{类别名,x_center,y_center,w,h,置信度}   实际上这个置信度是yolo网络输出的置信度confidence和预测对应的类别概率的乘积
        """
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.num_class))  # [7,7,2,20]
        class_probs = np.reshape(
            output[0:self.boundary1],
            (self.cell_size, self.cell_size, self.num_class))  # 类别概率 [7,7,20]
        scales = np.reshape(
            output[self.boundary1:self.boundary2],
            (self.cell_size, self.cell_size, self.boxes_per_cell))  # 置信度 [7,7,2]
        boxes = np.reshape(
            output[self.boundary2:],
            (self.cell_size, self.cell_size, self.boxes_per_cell, 4))  # 边界框 [7,7,2,4]
        offset = np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell)  # [14,7]  每一行[0,1,2,3,4,5,6]
        offset = np.transpose(
            np.reshape(
                offset,
                [self.boxes_per_cell, self.cell_size, self.cell_size]),
            (1, 2, 0))  # [7,7,2] 每一行都是  [[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]]

        boxes[:, :, :, 0] += offset  # 目标中心是相对于整个图片的
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size  # 宽度、高度相对整个图片的
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        # 转换成实际的编辑框(没有归一化的)
        boxes *= self.image_size

        # 遍历每一个边界框的置信度
        for i in range(self.boxes_per_cell):
            # 遍历每一个类别
            for j in range(self.num_class):
                # 在测试时，乘以条件类概率和单个盒子的置信度预测，这些分数编码了j类出现在框i中的概率以及预测框拟合目标的程度。
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])

        # [7,7,2,20] 如果第i个边界框检测到类别j 则[;,;,i,j]=1
        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        # 返回filter_mat_probs非0值的索引 返回4个List，每个list长度为n  即检测到的边界框的个数
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        # 获取检测到目标的边界框 [n,4]  n表示边界框的个数
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
        # 获取检测到目标的边界框的置信度 (n,)
        probs_filtered = probs[filter_mat_probs]
        # 获取检测到目标的边界框对应的目标类别 (n,)
        classes_num_filtered = np.argmax(
            filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        # 按置信度倒序排序，返回对应的索引
        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                # 计算n各边界框，两两之间的IoU是否大于阈值，非极大值抑制
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        # 非极大值抑制后的输出
        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        # 遍历每一个边界框
        for i in range(len(boxes_filtered)):
            result.append(
                [self.classes[classes_num_filtered[i]],  # 类别名
                 boxes_filtered[i][0],  # x中心
                 boxes_filtered[i][1],  # y中心
                 boxes_filtered[i][2],  # 宽度
                 boxes_filtered[i][3],  # 高度
                 probs_filtered[i]])  # 置信度

        return result

    def iou(self, box1, box2):
        """
        计算两个边界框的IoU

        :param box1: 边界框1  [4,]   真实值
        :param box2: 边界框2  [4,]   真实值
        :return:
        """
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
             max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
             max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

    def camera_detector(self, wait=10):
        """
        打开摄像头实时监测
        """
        detect_timer = Timer()  # 测试时间
        cap = cv2.VideoCapture(0)
        ret, _ = cap.read()  # 读取一帧

        while ret:
            ret, frame = cap.read()  # 读取一帧
            detect_timer.tic()  # 测试其实时间
            result = self.detect(frame)  # 测试结束时间
            detect_timer.toc()
            print('Average detecting time: {:.3f}s'.format(
                detect_timer.average_time))

            self.draw_result(frame, result)  # 绘制边界框，以及添加附加信息
            cv2.imshow('Camera', frame)  # 显示
            cv2.waitKey(wait)

            ret, frame = cap.read()

    def image_detector(self, imname, wait=0):
        """
        目标检测

        :param imname: 测试图片路径
        :param wait:
        """
        detect_timer = Timer()  # 检测时间
        image = cv2.imread(imname)  # 读取图片

        detect_timer.tic()  # 检测的起始时间
        result = self.detect(image)  # 开始检测
        detect_timer.toc()  # 检测的结束时间
        print('Average detecting time: {:.3f}s'.format(
            detect_timer.average_time))

        self.draw_result(image, result)  # 绘制检测结果
        cv2.imshow('Image', image)
        cv2.waitKey(wait)


def main():
    # 创建一个解析器对象，并告诉它将会有些什么参数。当程序运行时，该解析器就可以用于处理命令行参数。
    parser = argparse.ArgumentParser()
    # 定义参数
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', default='', type=str)
    # 定义了所有参数之后，你就可以给 parse_args() 传递一组参数字符串来解析命令行。默认情况下，参数是从 sys.argv[1:] 中获取
    # parse_args() 的返回值是一个命名空间，包含传递给命令的参数。该对象将参数保存其属性
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 设置环境变量

    yolo = YOLONet(False)  # 创建YOLO网络对象
    weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)  # 加载检查点文件
    detector = Detector(yolo, weight_file)  # 创建测试对象

    # detect from camera
    # cap = cv2.VideoCapture(-1)
    # detector.camera_detector(cap)

    # detect from image file
    # imname = 'test/person.jpg'
    # detector.image_detector(imname)
    detector.camera_detector()


if __name__ == '__main__':
    main()
