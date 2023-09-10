import csv
import random
from feature_batch import Random_embedding, Glove_embedding
from comparison_plot_batch import NN_embedding_plot
import platform

# 数据读入
data_on_windows = "D:/resources/datasets/sentiment-analysis-on-movie-reviews/train.tsv"
data_on_linux = "../../datasets/sentiment-analysis-on-movie-reviews/train.tsv"
os_name = platform.system()
with open(data_on_windows if os_name == "Windows" else data_on_linux) as f:
    tsvreader = csv.reader(f, delimiter='\t')
    temp = list(tsvreader)

embed_on_windows = "D:/resources/embeddings/glove.6B/glove.6B.50d"
embed_on_linux = "/shared_data/pretrained_models/glove.6B.50d.txt"
with open(embed_on_windows if os_name == "Windows" else embed_on_linux,'rb') as f:  # for glove embedding
    lines = f.readlines()

# 用GloVe创建词典
trained_dict = dict()
n = len(lines)
for i in range(n):
    line = lines[i].split()
    trained_dict[line[0].decode("utf-8").upper()] = [float(line[j]) for j in range(1, 51)]

# 初始化
iter_times = 50  # 做50个epoch
alpha = 0.001

# 程序开始
data = temp[1:]
batch_size = 500

# 随机初始化
random.seed(2021)
random_embedding = Random_embedding(data=data)
random_embedding.get_words()  # 找到所有单词，并标记ID
random_embedding.get_id()  # 找到每个句子拥有的单词ID

# 预训练模型初始化
random.seed(2021)
glove_embedding = Glove_embedding(data=data, trained_dict=trained_dict)
glove_embedding.get_words()  # 找到所有单词，并标记ID
glove_embedding.get_id()  # 找到每个句子拥有的单词ID

NN_embedding_plot(random_embedding, glove_embedding, alpha, batch_size, iter_times)
