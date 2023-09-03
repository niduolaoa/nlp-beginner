import numpy
import random
 
 
class Softmax:
    """Softmax regression"""
    def __init__(self, sample, typenum, feature):
        self.sample = sample  # 训练集样本个数
        self.typenum = typenum  # （情感）种类个数
        self.feature = feature  # 0-1向量的长度
        self.W = numpy.random.randn(feature, typenum)  # 参数矩阵W初始化，randn函数返回一个或一组样本，具有标准正态分布。返回值为指定维度的array
 
    def softmax_calculation(self, x):
        """x是向量，计算softmax值"""
        exp = numpy.exp(x - numpy.max(x))  # 先减去最大值防止指数太大溢出
        return exp / exp.sum()
 
    def softmax_all(self, wtx):
        """wtx是矩阵，即许多向量叠在一起，按行计算softmax值"""
        wtx -= numpy.max(wtx, axis=1, keepdims=True) # 先减去行最大值防止指数太大溢出
        wtx = numpy.exp(wtx)
        wtx /= numpy.sum(wtx, axis=1, keepdims=True)
        return wtx
 
    def change_y(self, y):
        """把（情感）种类转换为一个one-hot向量"""
        ans = numpy.array([0] * self.typenum)
        ans[y] = 1
        return ans.reshape(-1, 1)
 
    def prediction(self, X):
        """给定0-1矩阵X，计算每个句子的y_hat值（概率）"""
        prob = self.softmax_all(X.dot(self.W))
        return prob.argmax(axis=1)
 
    def correct_rate(self, train, train_y, test, test_y):
        """计算训练集和测试集的准确率"""
        # train set
        n_train = len(train)
        pred_train = self.prediction(train)
        train_correct = sum([train_y[i] == pred_train[i] for i in range(n_train)]) / n_train
        # test set
        n_test = len(test)
        pred_test = self.prediction(test)
        test_correct = sum([test_y[i] == pred_test[i] for i in range(n_test)]) / n_test
        print(train_correct, test_correct)
        return train_correct, test_correct
 
    def regression(self, X, y, alpha, times, strategy="mini", batch_size=100):
        """Softmax regression"""
        if self.sample != len(X) or self.sample != len(y):
            raise Exception("Sample size does not match!")  # 样本个数不匹配
        if strategy == "mini":  # mini-batch
            for i in range(times):
                increment = numpy.zeros((self.feature, self.typenum))  # 梯度初始为0矩阵
                for j in range(batch_size):  # 随机抽K次
                    k = random.randint(0, self.sample - 1)
                    yhat = self.softmax_calculation(self.W.T.dot(X[k].reshape(-1, 1)))
                    increment += X[k].reshape(-1, 1).dot((yhat - self.change_y(y[k])).T)  # 梯度加和
                self.W -= alpha / batch_size * increment  # 参数更新
        elif strategy == "shuffle":  # 随机梯度
            for i in range(times):
                k = random.randint(0, self.sample - 1)  # 每次抽一个
                yhat = self.softmax_calculation(self.W.T.dot(X[k].reshape(-1, 1)))
                increment = X[k].reshape(-1, 1).dot((yhat - self.change_y(y[k])).T)  # 计算梯度
                self.W -= alpha * increment  # 参数更新
        elif strategy=="batch":  # 整批量梯度
            for i in range(times):
                increment = numpy.zeros((self.feature, self.typenum))  ## 梯度初始为0矩阵
                for j in range(self.sample):  # 所有样本都要计算
                    yhat = self.softmax_calculation(self.W.T.dot(X[j].reshape(-1, 1)))
                    increment += X[j].reshape(-1, 1).dot((yhat - self.change_y(y[j])).T)  # 梯度加和
                self.W -= alpha / self.sample * increment  # 参数更新
        else:
            raise Exception("Unknown strategy")