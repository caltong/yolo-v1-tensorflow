import os
import argparse
import datetime
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc import pascal_voc

slim = tf.contrib.slim

"""
训练YOLO网络模型
"""


class Solver(object):
    """
    求解器的类，用于训练YOLO网络
    """

    def __init__(self, net, data):
        """
        构造函数，加载训练参数

        :param net: YOLONet对象
        :param data: pascal_voc对象
        """
        self.net = net  # yolo网络
        self.data = data  # voc2007数据处理
        self.weights_file = cfg.WEIGHTS_FILE  # 检查点文件路径
        self.max_iter = cfg.MAX_ITER  # 训练最大迭代次数
        self.initial_learning_rate = cfg.LEARNING_RATE  # 初始学习率
        self.decay_steps = cfg.DECAY_STEPS  # 退化学习率衰减步数
        self.decay_rate = cfg.DECAY_RATE  # 衰减率
        self.staircase = cfg.STAIRCASE  # 日志文件保存间隔步
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER  # 模型保存间隔步
        self.output_dir = os.path.join(
            cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))  # 输出文件夹路径
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg()  # 保存配置信息

        self.variable_to_restore = tf.global_variables()  # 创建变量，保存当前迭代次数
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)  # 退化学习率
        self.ckpt_file = os.path.join(self.output_dir, 'yolo')  # 指定保存的模型名称
        self.summary_op = tf.summary.merge_all()  # 合并所有的summary
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)  # 创建writer，指定日志文件路径，用于写日志文件

        self.global_step = tf.train.create_global_step()  # 创建变量，保存当前迭代次数
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')  # 退化学习率
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)  # 创建求解器
        self.train_op = slim.learning.create_train_op(
            self.net.total_loss, self.optimizer, global_step=self.global_step)

        gpu_options = tf.GPUOptions()  # 设置GPU使用资源
        config = tf.ConfigProto(gpu_options=gpu_options)  # 按需分配GPU使用的资源
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())  # 运行图之前，初始化变量

        # 恢复模型
        if self.weights_file is not None:
            print('Restoring weights from: ' + self.weights_file)
            self.saver.restore(self.sess, self.weights_file)

        self.writer.add_graph(self.sess.graph)  # 将图写入日志文件

    def train(self):
        """
        开始训练
        """

        train_timer = Timer()  # 训练时间
        load_timer = Timer()  # 数据集加载时间

        # 开始迭代
        for step in range(1, self.max_iter + 1):

            load_timer.tic()  # 计算每次迭代加载数据的起始时间
            images, labels = self.data.get()  # 加载数据集 每次读取batch大小的图片以及图片对应的标签
            load_timer.toc()  # 计算这次迭代加载数据集所使用的时间
            feed_dict = {self.net.images: images,
                         self.net.labels: labels}

            # 迭代summary_iter次，保存一次日志文件，迭代summary_iter*10次，输出一次的迭代信息
            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:

                    train_timer.tic()  # 计算每次迭代训练的起始时间
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.total_loss, self.train_op],
                        feed_dict=feed_dict)  # 开始迭代训练，每一次迭代后global_step自加1
                    train_timer.toc()

                    # 输出信息
                    log_str = '{} Epoch: {}, Step: {}, Learning rate: {},Loss: {:5.3f}\nSpeed: {:.3f}s/iter,' \
                              'Load: {:.3f}s/iter, Remain: {}'.format(
                                datetime.datetime.now(),
                                self.data.epoch,
                                int(step),
                                round(self.learning_rate.eval(session=self.sess), 6),
                                loss,
                                train_timer.average_time,
                                load_timer.average_time,
                                train_timer.remain(step, self.max_iter))
                    print(log_str)

                else:
                    train_timer.tic()  # 计算每次迭代训练的起始时间
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)  # 开始迭代训练，每一次迭代后global_step自加1
                    train_timer.toc()  # 计算这次迭代训练所使用的时间

                self.writer.add_summary(summary_str, step)  # 将summary写入文件

            else:
                train_timer.tic()  # 计算每次迭代训练的起始时间
                self.sess.run(self.train_op, feed_dict=feed_dict)  # 开始迭代训练，每一次迭代后global_step自加1
                train_timer.toc()  # 计算这次迭代训练所使用的时间

            # 没迭代save_iter次，保存一次模型
            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(
                    self.sess, self.ckpt_file, global_step=self.global_step)

    def save_cfg(self):
        """
        保存配置信息
        """

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)


def update_config_paths(data_dir, weights_file):
    """
    数据集路径，和模型检查点文件路径

    :param data_dir: 数据文件夹  数据集放在pascal_voc目录下
    :param weights_file: 检查点文件名 该文件放在数据集目录下的weights文件夹下
    :return:
    """
    cfg.DATA_PATH = data_dir
    cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')  # 数据所在文件夹
    cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')  # VOC2007数据所在文件夹
    cfg.OUTPUT_DIR = os.path.join(cfg.PASCAL_PATH, 'output')  # 保存生成的数据集标签缓冲文件所在文件夹
    cfg.WEIGHTS_DIR = os.path.join(cfg.PASCAL_PATH, 'weights')  # 保存生成的网络模型和日志文件所在的文件夹
    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)  # 检查点文件所在的目录


def main():
    # 创建一个解析器对象，并告诉它将会有些什么参数。当程序运行时，该解析器就可以用于处理命令行参数。
    parser = argparse.ArgumentParser()  # 定义参数
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)  # 权重文件名
    parser.add_argument('--data_dir', default="data", type=str)  # 数据集路径
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default='', type=str)
    # 定义了所有参数之后，你就可以给 parse_args() 传递一组参数字符串来解析命令行。默认情况下，参数是从 sys.argv[1:] 中获取
    # parse_args() 的返回值是一个命名空间，包含传递给命令的参数。该对象将参数保存其属性
    args = parser.parse_args()

    # 判断是否是使用gpu
    if args.gpu is not None:
        cfg.GPU = args.gpu

    # 设定数据集路径，以及检查点文件路径
    if args.data_dir != cfg.DATA_PATH:
        update_config_paths(args.data_dir, args.weights)

    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo = YOLONet()  # 创建YOLO网络对象
    pascal = pascal_voc('train')  # 数据集对象

    solver = Solver(yolo, pascal)  # 求解器对象

    print('Start training ...')
    solver.train()  # 开始训练
    print('Done training.')


if __name__ == '__main__':
    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()
