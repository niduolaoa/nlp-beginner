import numpy
import csv
import random
from feature import Bag,Gram
from compassion import alpha_gradient_plot
import platform
 
# 数据读取
data_on_windows = "D:/resources/datasets/sentiment-analysis-on-movie-reviews/train.tsv"
data_on_linux = "../../datasets/sentiment-analysis-on-movie-reviews/train.tsv"
os_name = platform.system()
with open(data_on_windows if os_name == "Windows" else data_on_linux) as f:
    tsvreader = csv.reader(f, delimiter='\t')
    temp = list(tsvreader)
 
# 初始化
data = temp[1:]
max_item=1000
random.seed(2023)
numpy.random.seed(2023)
 
# 特征提取
bag=Bag(data,max_item)
bag.get_words()
bag.get_matrix()
 
gram=Gram(data, dimension=2, max_item=max_item)
gram.get_words()
gram.get_matrix()
 
# 画图
alpha_gradient_plot(bag, gram,10000, 10)  # 计算10000次
alpha_gradient_plot(bag, gram,100000, 10)  # 计算100000次