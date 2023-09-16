from feature import get_batch, Word_Embedding
from torch import optim
import random, numpy, torch
from neural_network import Language
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import math


random.seed(2021)
numpy.random.seed(2021)
torch.cuda.manual_seed(2021)
torch.manual_seed(2021)

with open('../../datasets/poetryFromTang.txt', 'rb') as f:
    temp = f.readlines()

a = Word_Embedding(temp)
a.data_process()
train = get_batch(a.matrix, 1)
learning_rate = 0.004
iter_times = 10

strategies = ['lstm', 'gru']
train_perplexity_records = list()
models = list()
for i in range(2):
    model = Language(50, len(a.word_dict), 50, a.tag_dict, a.word_dict, strategy=strategies[i])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = F.cross_entropy
    train_perplexity_record = list()
    model = model.cuda()
    for iteration in range(iter_times):
        model.train()
        log_probs = []
        for j, batch in enumerate(train):
            x = batch.cuda()
            x, y = x[:,:-1], x[:,1:]        # y的形状是[batch_size, sequence_length]
            pred = model(x).transpose(1,2)  # pred维度转置后的形状是[batch_size, len_words， sequence_length]
            optimizer.zero_grad()
            loss = criterion(pred, y)  # pred和y的最后一个维度相同，形状匹配，y不需要额外进行独热编码
            log_prob = loss.item()
            log_probs.append(log_prob)
            loss.backward()
            optimizer.step()
        perplexity = math.exp(sum(log_probs) / len(train.dataset))  # 困惑度可以用来衡量两个分布之间差异
        print("---------- Iteration", iteration + 1, "----------")
        print("Train perplexity:", perplexity)
        train_perplexity_record.append(perplexity)
    train_perplexity_records.append(train_perplexity_record)
    models.append(model)


def cat_poem(l):
    """拼接诗句"""
    poem = list()
    for item in l:
        poem.append(''.join(item))
    return poem

# 选择训练的模型: ['lstm', 'gru']
model = models[0]
# 生成固定诗句
poem = cat_poem(model.generate_random_poem(16, 4, random=False))
for sent in poem:
    print(sent)

# 生成随机诗句
torch.manual_seed(2021)
poem = cat_poem(model.generate_random_poem(12, 6, random=True))
for sent in poem:
    print(sent)

# 生成固定藏头诗
poem = cat_poem(model.generate_hidden_head("春夏秋冬", max_len=20, random=False))
for sent in poem:
    print(sent)

# 生成随机藏头诗
torch.manual_seed(0)
poem=cat_poem(model.generate_hidden_head("春夏秋冬", max_len=20, random=True))
for sent in poem:
    print(sent)

# 画图
save_folder = "./result/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

x = list(range(1, iter_times + 1))
plt.plot(x, train_perplexity_records[0], 'r--',label='Lstm')
plt.plot(x, train_perplexity_records[1], 'g--',label='Gru')
plt.legend()
plt.title("Train Perplexity")
plt.xlabel("Iterations")
plt.ylabel("Perplexity")
plt.savefig(save_folder + 'perplexity.jpg')