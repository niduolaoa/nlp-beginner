import numpy
import random
 
 
def data_split(data, test_rate=0.3, max_item=1000):
    """把数据按一定比例划分成训练集和测试集"""
    train = list()
    test = list()
    i = 0
    for datum in data:
        i += 1
        if random.random() > test_rate:
            train.append(datum)
        else:
            test.append(datum)
        if i > max_item:
            break
    return train, test
 
 
class Bag:
    """Bag of words"""
    def __init__(self, my_data, max_item=1000):
        self.data = my_data[:max_item]
        self.max_item = max_item
        self.dict_words = dict()  # 单词到单词编号的映射
        self.len = 0  # 记录有几个单词
        self.train, self.test = data_split(my_data, test_rate=0.3, max_item=max_item)
        self.train_y = [int(term[3]) for term in self.train]  # 训练集类别
        self.test_y = [int(term[3]) for term in self.test]  # 测试集类别
        self.train_matrix = None  # 训练集的0-1矩阵（每行一个句子）
        self.test_matrix = None  # 测试集的0-1矩阵（每行一个句子）
 
    def get_words(self):
        for term in self.data:
            s = term[2]
            s = s.upper()  # 记得要全部转化为大写！！（或者全部小写，否则一个单词例如i，I会识别成不同的两个单词）
            words = s.split()  # 分割
            for word in words:  # 一个一个单词寻找
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words)
        self.len = len(self.dict_words)
        self.test_matrix = numpy.zeros((len(self.test), self.len))  # 初始化0-1矩阵
        self.train_matrix = numpy.zeros((len(self.train), self.len))  # 初始化0-1矩阵
 
    def get_matrix(self):
        for i in range(len(self.train)):  # 训练集矩阵
            s = self.train[i][2]
            words = s.split()
            for word in words:
                word = word.upper()
                self.train_matrix[i][self.dict_words[word]] = 1
        for i in range(len(self.test)):  # 测试集矩阵
            s = self.test[i][2]
            words = s.split()
            for word in words:
                word = word.upper()
                self.test_matrix[i][self.dict_words[word]] = 1
 
 
class Gram:
    """N-gram"""
    def __init__(self, my_data, dimension=2, max_item=1000):
        self.data = my_data[:max_item]
        self.max_item = max_item
        self.dict_words = dict()  # 特征到t正编号的映射
        self.len = 0  # 记录有多少个特征
        self.dimension = dimension  # 决定使用几元特征
        self.train, self.test = data_split(my_data, test_rate=0.3, max_item=max_item)
        self.train_y = [int(term[3]) for term in self.train]  # 训练集类别
        self.test_y = [int(term[3]) for term in self.test]  # 测试集类别
        self.train_matrix = None  # 训练集0-1矩阵（每行代表一句话）
        self.test_matrix = None  # 测试集0-1矩阵（每行代表一句话）
 
    def get_words(self):
        for d in range(1, self.dimension + 1):  # 提取 1-gram, 2-gram,..., N-gram 特征
            for term in self.data:
                s = term[2]
                s = s.upper()  # 记得要全部转化为大写！！（或者全部小写，否则一个单词例如i，I会识别成不同的两个单词）
                words = s.split()
                for i in range(len(words) - d + 1):  # 一个一个特征找
                    temp = words[i:i + d]
                    temp = "_".join(temp)  # 形成i d-gram 特征
                    if temp not in self.dict_words:
                        self.dict_words[temp] = len(self.dict_words)
        self.len = len(self.dict_words)
        self.test_matrix = numpy.zeros((len(self.test), self.len))  # 训练集矩阵初始化
        self.train_matrix = numpy.zeros((len(self.train), self.len))  # 测试集矩阵初始化
 
    def get_matrix(self):
        for d in range(1, self.dimension + 1):
            for i in range(len(self.train)):  # 训练集矩阵
                s = self.train[i][2]
                s = s.upper()
                words = s.split()
                for j in range(len(words) - d + 1):
                    temp = words[j:j + d]
                    temp = "_".join(temp)
                    self.train_matrix[i][self.dict_words[temp]] = 1
            for i in range(len(self.test)):  # 测试集矩阵
                s = self.test[i][2]
                s = s.upper()
                words = s.split()
                for j in range(len(words) - d + 1):
                    temp = words[j:j + d]
                    temp = "_".join(temp)
                    self.test_matrix[i][self.dict_words[temp]] = 1