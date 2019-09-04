import os

"""
path and dataset parameter
路径和数据集参数
"""

DATA_PATH = 'data'  # 所有数据所在根目录

PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')  # VOC2012数据集所在的目录

CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')  # 保存生成的数据集标签缓冲文件所在文件夹

OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')  # 保存生成的网络模型和日志所在文件夹

WEIGHTS_DIR = os.path.join(DATA_PATH, 'weights')  # 检查点文件所在目录

WEIGHTS_META = 'YOLO.meta'
# WEIGHTS_FILE = None
WEIGHTS_FILE = os.path.join(DATA_PATH, 'weights', 'YOLO_small.ckpt')  # /data/weight/YOLO_small.ckpt

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']  # 数据集类别名称

FLIPPED = True  # 使用水平镜像，扩大一倍数据集？

"""
model parameter
网络模型参数
"""

IMAGE_SIZE = 448  # 图片大小

CELL_SIZE = 7  # grid cell 数量

BOXES_PER_CELL = 2  # 单个grid cell box数量

ALPHA = 0.1  # leaky relu 系数

DISP_CONSOLE = False  # 控制台输出信息

# 损失函数 权重设置
OBJECT_SCALE = 1.0  # 有目标时 置信度权重
NOOBJECT_SCALE = 1.0  # 没目标时 置信度权重
CLASS_SCALE = 2.0  # 类别权重
COORD_SCALE = 5.0  # bounding box 坐标权重

"""
solver parameter
训练参数设置
"""

GPU = ''

LEARNING_RATE = 0.0001  # 学习率

DECAY_STEPS = 30000  # 学习率衰减步数

DECAY_RATE = 0.1  # 衰减率

STAIRCASE = True

BATCH_SIZE = 45  # 批数量

MAX_ITER = 15000  # 最大迭代次数

SUMMARY_ITER = 10  # 日志保存间隔

SAVE_ITER = 1000  # 模型保存间隔

"""
test parameter
测试相关参数
"""

THRESHOLD = 0.2  # cell 有目标的置信度阈值

IOU_THRESHOLD = 0.5  # 非极大抑制 IOU阈值
