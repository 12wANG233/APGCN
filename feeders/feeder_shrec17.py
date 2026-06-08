# 导入必要的库
import json  # 用于处理JSON格式的数据
import numpy as np  # 用于数值计算和数组操作
import random  # 用于生成随机数
from torch.utils.data import Dataset  # PyTorch数据集基类
import torch  # PyTorch深度学习框架


class Feeder(Dataset):
    """
    SHREC17手部骨架数据加载器，继承自PyTorch的Dataset类
    用于加载和处理手部骨架序列数据，支持多种数据增强和预处理技术
    """

    def __init__(self, data_path, label_path=None, p_interval=1, num_clips=1, split='train', random_choose=False,
                 random_shift=False, random_move=False, random_rot=False, frame_sample='resize', align=False,
                 spatial_flip=False, drop_joint=False, drop_axis=False, window_size=-1, normalization=False,
                 debug=False, use_mmap=False, bone=False, vel=False, label_flag=28):
        """
        初始化数据加载器，设置各种数据处理参数

        参数说明:
        :param data_path: 字符串，数据文件所在的根目录路径
        :param label_path: 字符串，可选，标签文件路径
        :param p_interval: 整数，概率间隔，默认为1
        :param num_clips: 整数，剪辑数量，默认为1
        :param split: 字符串，数据集分割类型，'train'或'val'，默认为'train'
        :param random_choose: 布尔值，是否随机选择输入序列的一部分，默认为False
        :param random_shift: 布尔值，是否在序列的开始或结束随机填充零，默认为False
        :param random_move: 布尔值，是否应用随机移动增强，默认为False
        :param random_rot: 布尔值，是否围绕xyz轴随机旋转骨架，默认为False
        :param frame_sample: 字符串，帧采样方法，默认为'resize'
        :param align: 布尔值，是否对齐数据，默认为False
        :param spatial_flip: 布尔值，是否进行空间翻转，默认为False
        :param drop_joint: 布尔值，是否丢弃关节，默认为False
        :param drop_axis: 布尔值，是否丢弃轴，默认为False
        :param window_size: 整数，输出序列的长度，-1表示使用原始长度，默认为-1
        :param normalization: 布尔值，是否归一化输入序列，默认为False
        :param debug: 布尔值，调试模式，仅使用前100个样本，默认为False
        :param use_mmap: 布尔值，是否使用mmap模式加载数据以节省内存，默认为False
        :param bone: 布尔值，是否使用骨骼模态，默认为False
        :param vel: 布尔值，是否使用速度模态，默认为False
        :param label_flag: 整数，标签类型（14或28类），默认为28
        """
        # 设置数据集根目录路径
        self.nw_hand17_root = data_path

        # 确定是训练集还是验证集
        if split == 'test' or (label_path and 'test' in label_path):
            # 验证集/测试集设置
            self.split = 'test'
            # 加载测试集样本信息JSON文件
            with open(self.nw_hand17_root + 'test_samples.json', 'r') as f1:
                json_file = json.load(f1)  # 解析JSON内容
            self.data_dict = json_file  # 存储样本信息字典
            self.flag = 'test_jsons/'  # 设置测试集数据子目录标识
        else:
            # 训练集设置
            self.split = 'train'
            # 加载训练集样本信息JSON文件
            with open(self.nw_hand17_root + 'train_samples.json', 'r') as f2:
                json_file = json.load(f2)  # 解析JSON内容
            self.data_dict = json_file  # 存储样本信息字典
            self.flag = 'train_jsons/'  # 设置训练集数据子目录标识

        # 定义手部骨骼连接关系（21个关节点之间的连接）
        # 每个元组表示一个骨骼连接，格式为(子关节, 父关节)
        self.bone_pairs = [(1, 2), (3, 1), (4, 3), (5, 4), (6, 5), (7, 2), (8, 7), (9, 8), (10, 9),
                           (11, 2), (12, 11), (13, 12), (14, 13), (15, 2), (16, 15), (17, 16),
                           (18, 17), (19, 2), (20, 19), (21, 20), (22, 21)]

        # 存储所有初始化参数到实例变量中
        self.debug = debug  # 调试模式标志
        self.data_path = data_path  # 数据路径
        self.label_path = label_path  # 标签路径
        self.split = split  # 数据集分割类型
        self.random_choose = random_choose  # 随机选择标志
        self.random_shift = random_shift  # 随机偏移标志
        self.random_move = random_move  # 随机移动标志
        self.random_rot = random_rot  # 随机旋转标志
        self.window_size = window_size  # 窗口大小
        self.num_clips = num_clips  # 剪辑数量
        self.normalization = normalization  # 归一化标志
        self.use_mmap = use_mmap  # 内存映射标志
        self.p_interval = p_interval  # 概率间隔
        self.align = align  # 对齐标志
        self.frame_sample = frame_sample  # 帧采样方法
        self.spatial_flip = spatial_flip  # 空间翻转标志
        self.drop_joint = drop_joint  # 丢弃关节标志
        self.drop_axis = drop_axis  # 丢弃轴标志
        self.bone = bone  # 骨骼模态标志
        self.vel = vel  # 速度模态标志
        self.label_flag = label_flag  # 标签类型标志

        # 根据标签类型设置类别数量
        self.num_classes = 14 if label_flag == 14 else 28

        # 加载数据
        self.load_data()

        # 如果需要归一化，计算数据的均值和标准差
        if normalization:
            self.get_mean_map()

        # 验证所有标签都在有效范围内
        self.validate_labels()

        # 为每个样本生成唯一的名称标识
        self.sample_name = [f'{self.split}_{i}' for i in range(len(self.data))]

    def validate_labels(self):
        """
        验证所有标签都在有效范围内
        如果发现无效标签，会打印警告并将标签映射到有效范围内
        """
        # 创建有效标签集合
        valid_labels = set(range(self.num_classes))

        # 遍历所有标签
        for i, label in enumerate(self.label):
            if label not in valid_labels:
                # 打印警告信息
                print(f"警告: 索引 {i} 的标签 {label} 超出有效范围 [0, {self.num_classes - 1}]")
                # 将无效标签映射到有效范围（取模运算）
                self.label[i] = label % self.num_classes

    def load_data(self):
        """
        加载骨架数据和对应标签
        从JSON文件中读取骨架数据并转换为合适的格式
        """
        self.data = []  # 存储骨架数据的列表
        self.label = []  # 存储标签的列表

        # 遍历数据字典中的所有样本
        for data_item in self.data_dict:
            # 获取文件名
            file_name = data_item['file_name']

            # 打开并读取JSON文件
            with open(self.nw_hand17_root + self.flag + file_name + '.json', 'r') as f:
                json_file = json.load(f)  # 解析JSON内容

            # 提取骨架数据
            skeletons = json_file['skeletons']
            # 转换为NumPy数组，指定数据类型为float32
            value = np.array(skeletons, dtype=np.float32)

            # 转换为CTVM格式 (通道, 时间, 关节, 人数)
            # 原始数据格式为(T, V, C) -> 转置为(C, T, V)
            value = value.transpose(2, 0, 1)
            # 添加人数维度(M)，扩展数组维度
            value = np.expand_dims(value, axis=-1)

            # 将处理后的数据添加到数据列表
            self.data.append(value)

            # 根据label_flag参数选择标签类型
            if self.label_flag == 14:
                # 使用14类标签，并转换为0-indexed
                label_val = int(data_item['label_14']) - 1
            elif self.label_flag == 28:
                # 使用28类标签，并转换为0-indexed
                label_val = int(data_item['label_28']) - 1
            else:
                # 默认使用28类标签，并转换为0-indexed
                label_val = int(data_item['label_28']) - 1
                # 打印警告信息
                print(f"警告: 未知的label_flag值 {self.label_flag}，使用默认值28")

            # 确保标签在有效范围内
            label_val = max(0, min(label_val, self.num_classes - 1))
            # 将标签添加到标签列表
            self.label.append(label_val)

        # 如果处于调试模式，只使用前100个样本
        if self.debug:
            self.data = self.data[:100]
            self.label = self.label[:100]

    def get_mean_map(self):
        """
        计算数据的均值和标准差用于归一化
        仅在normalization参数为True时调用
        """
        # 将所有数据沿着时间轴拼接
        all_data = np.concatenate([d for d in self.data], axis=1)
        # 计算均值，保持维度以便广播
        self.mean_map = np.mean(all_data, axis=(1, 2, 3), keepdims=True)
        # 计算标准差，保持维度以便广播
        self.std_map = np.std(all_data, axis=(1, 2, 3), keepdims=True)

        # 避免除以零：将标准差为零的位置替换为1.0
        self.std_map = np.where(self.std_map == 0, 1.0, self.std_map)

    def random_translation(self, ske_data):
        translate = np.eye(3)
        random.random()
        t_x = random.uniform(-0.01, 0.01)
        t_y = random.uniform(-0.01, 0.01)
        t_z = random.uniform(-0.01, 0.01)

        translate[0, 0] = translate[0, 0] + t_x
        translate[1, 1] = translate[1, 1] + t_y
        translate[2, 2] = translate[2, 2] + t_z

        data = np.dot(ske_data, translate)

        return data

    def __len__(self):

        return len(self.label)

    def __iter__(self):

        return self

    def __getitem__(self, index):

        data_numpy = self.data[index].copy()

        # data_numpy = self.random_translation(data_numpy)


        label = self.label[index]
        label = int(label)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map

        valid_frame_num = np.sum(np.any(data_numpy != 0, axis=(0, 2, 3)))
        if self.frame_sample == 'resize' and self.window_size > 0:
            data_numpy = self.valid_crop_resize(data_numpy, valid_frame_num)

        if self.random_rot and self.split == 'train':
            data_numpy = self.random_rotation(data_numpy)

        if self.random_move and self.split == 'train':
            data_numpy = self.random_translation(data_numpy)


        modalities = [data_numpy]  # 基础关节模态
        # modalities = [] 


        vel_numpy = np.zeros_like(data_numpy)
        if data_numpy.shape[1] > 1:
            vel_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
        modalities.append(vel_numpy)

        bone_numpy = np.zeros_like(data_numpy)
        for v1, v2 in self.bone_pairs:
            if v1 <= bone_numpy.shape[2] and v2 <= bone_numpy.shape[2]:
                bone_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
        modalities.append(bone_numpy)


        # shift_numpy = np.zeros_like(data_numpy)
        # for i in range(data_numpy.shape[0]):
        #     for v1, v2 in self.bone_pairs:
        #         if v1 <= shift_numpy.shape[2] and v2 <= shift_numpy.shape[2]:
        #             shift_numpy[i-1, :, v1 - 1] = data_numpy[i, :, v1 - 1] - data_numpy[i-2, :, v2 - 1]
        # modalities.append(shift_numpy)

        if len(modalities) > 1:
            data_numpy = np.concatenate(modalities, axis=0)


        data_numpy = np.nan_to_num(data_numpy)
        data_numpy = np.clip(data_numpy, -10, 10)
        data_tensor = torch.from_numpy(data_numpy).float()


        return data_tensor, label, index

    def valid_crop_resize(self, data_numpy, valid_frame_num):
        """
        有效的帧裁剪和调整大小方法
        根据窗口大小调整数据的时间维度长度

        参数:
        data_numpy: numpy数组，输入骨架数据
        valid_frame_num: 整数，有效帧数

        返回:
        numpy数组: 处理后的骨架数据
        """
        # 获取数据的形状：通道数、时间长度、关节数、人数
        C, T, V, M = data_numpy.shape

        # 如果指定了窗口大小，调整到该大小
        if self.window_size > 0:
            # 如果当前时间长度小于窗口大小
            if T < self.window_size:
                # 创建填充后的零数组
                data_numpy_padded = np.zeros((C, self.window_size, V, M), dtype=np.float32)
                # 将原始数据复制到填充数组的前面部分
                data_numpy_padded[:, :T, :, :] = data_numpy
                # 使用填充后的数据
                data_numpy = data_numpy_padded
            else:
                # 当前时间长度大于窗口大小，需要进行裁剪
                if self.split == 'train':
                    # 训练集：随机选择起始位置
                    begin = random.randint(0, T - self.window_size)
                else:
                    # 验证集/测试集：选择中间位置
                    begin = (T - self.window_size) // 2
                # 裁剪数据
                data_numpy = data_numpy[:, begin:begin + self.window_size, :, :]

        # 返回处理后的数据
        return data_numpy

    def random_rotation(self, data):
        """
        应用随机旋转增强

        参数:
        data: numpy数组，输入骨架数据

        返回:
        numpy数组: 旋转后的骨架数据
        """
        # 生成随机旋转角度（-10度到10度之间）
        angle = random.uniform(-10, 10)
        # 将角度转换为弧度
        angle_rad = np.radians(angle)

        # 创建绕Z轴的旋转矩阵
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],  # X轴旋转分量
            [np.sin(angle_rad), np.cos(angle_rad), 0],  # Y轴旋转分量
            [0, 0, 1]  # Z轴保持不变
        ], dtype=np.float32)

        # 应用旋转矩阵
        # 首先调整数据维度顺序为(T, V, M, C)以便进行矩阵乘法
        data_rotated = np.dot(data.transpose(1, 2, 3, 0), rotation_matrix.T)
        # 恢复原始维度顺序(C, T, V, M)
        return data_rotated.transpose(3, 0, 1, 2)

    def random_translation(self, data):
        """
        应用随机平移增强

        参数:
        data: numpy数组，输入骨架数据

        返回:
        numpy数组: 平移后的骨架数据
        """
        # 创建单位矩阵作为基础变换矩阵
        translate = np.eye(3, dtype=np.float32)
        # 生成随机平移量（小范围平移）
        t_x = random.uniform(-0.01, 0.01)  # X轴平移
        t_y = random.uniform(-0.01, 0.01)  # Y轴平移
        t_z = random.uniform(-0.01, 0.01)  # Z轴平移

        # 将平移量添加到变换矩阵的对角线上
        translate[0, 0] = translate[0, 0] + t_x
        translate[1, 1] = translate[1, 1] + t_y
        translate[2, 2] = translate[2, 2] + t_z

        # 应用变换矩阵
        # 首先调整数据维度顺序为(T, V, M, C)以便进行矩阵乘法
        data_translated = np.dot(data.transpose(1, 2, 3, 0), translate.T)
        # 恢复原始维度顺序(C, T, V, M)
        return data_translated.transpose(3, 0, 1, 2)

    def top_k(self, score, top_k):
        """
        计算top-k准确率

        参数:
        score: numpy数组，预测得分矩阵
        top_k: 整数，top-k值

        返回:
        float: top-k准确率
        """
        # 对得分进行排序，获取索引
        rank = score.argsort()
        # 检查真实标签是否在top-k预测中
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        # 计算准确率
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    """
    动态导入类或函数

    参数:
    name: 字符串，类的完整路径（如"module.submodule.ClassName"）

    返回:
    类或函数对象
    """
    # 按点分割路径
    components = name.split('.')
    # 导入第一个组件（模块）
    mod = __import__(components[0])
    # 遍历剩余组件，逐步获取子模块或属性
    for comp in components[1:]:
        mod = getattr(mod, comp)
    # 返回最终的类或函数
    return mod