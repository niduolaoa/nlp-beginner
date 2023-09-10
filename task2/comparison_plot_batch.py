import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import optim
from neural_network_batch import MY_RNN,MY_CNN
from feature_batch import get_batch
import os


def NN_embdding(model, train, test, learning_rate, iter_times):
	# 定义优化器（求参数）
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 损失函数  
    loss_fun = F.cross_entropy
    # 损失值记录
    train_loss_record = list()
    test_loss_record = list()
    long_loss_record = list()
    # 准确率记录
    train_record = list()
    test_record = list()
    long_record = list()
    # torch.autograd.set_detect_anomaly(True)  # 如果发生梯度爆炸或梯度消失等数值异常情况，PyTorch会引发异常
	# 训练阶段
    for iteration in range(iter_times):
        model.train()  # 训练模式
        for i, batch in enumerate(train):
            x, y = batch  # 取一个batch
            y = y.cuda()
            pred = model(x).cuda()  # 计算输出
            optimizer.zero_grad()  # 梯度初始化
            loss = loss_fun(pred, y).cuda()  # 损失值计算
            loss.backward()  # 反向传播梯度
            optimizer.step()  # 更新参数

        model.eval()  # 推理模式
        # 本轮正确率记录
        train_acc = list()
        test_acc = list()
        long_acc = list()
        length = 20
        # 本轮损失值记录
        train_loss = 0
        test_loss = 0
        long_loss = 0
        for i, batch in enumerate(train):
            x, y = batch  # 取一个batch
            y = y.cuda()
            pred = model(x).cuda()  # 计算输出
            loss = loss_fun(pred, y).cuda()    # 损失值计算
            train_loss += loss.item()  # 损失值累加
            _, y_pre = torch.max(pred, -1)
            # 计算本batch准确率
            acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
            train_acc.append(acc)

        for i, batch in enumerate(test):
            x, y = batch  # 取一个batch
            y = y.cuda()
            pred = model(x).cuda()  # 计算输出
            loss = loss_fun(pred, y).cuda()  # 损失值计算
            test_loss += loss.item()  # 损失值累加
            _, y_pre = torch.max(pred, -1)
            # 计算本batch准确率
            acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
            test_acc.append(acc)
            if len(x[0]) > length:  # 长句子侦测
              long_acc.append(acc)
              long_loss += loss.item()

        trains_acc = sum(train_acc) / len(train_acc)
        tests_acc = sum(test_acc) / len(test_acc)
        longs_acc = sum(long_acc) / len(long_acc)
        
        train_loss_record.append(train_loss / len(train_acc))
        test_loss_record.append(test_loss / len(test_acc))
        long_loss_record.append(long_loss / len(long_acc))
        train_record.append(trains_acc.cpu())
        test_record.append(tests_acc.cpu())
        long_record.append(longs_acc.cpu())
        print("---------- Iteration", iteration + 1, "----------")
        print("Train loss:", train_loss / len(train_acc))
        print("Test loss:", test_loss / len(test_acc))
        print("Train accuracy:", trains_acc.cpu())
        print("Test accuracy:", tests_acc.cpu())
        print("Long sentence accuracy:", longs_acc.cpu())

    return train_loss_record, test_loss_record, long_loss_record, train_record, test_record, long_record


def NN_embedding_plot(random_embedding, glove_embedding, learning_rate, batch_size, iter_times):
	# 获得训练集和测试集的batch
    train_random = get_batch(random_embedding.train_matrix,
                             random_embedding.train_y, batch_size)
    test_random = get_batch(random_embedding.test_matrix,
                            random_embedding.test_y, batch_size)
    train_glove = get_batch(glove_embedding.train_matrix,
                            glove_embedding.train_y, batch_size)
    test_glove = get_batch(random_embedding.test_matrix,
                           glove_embedding.test_y, batch_size)
    # 模型建立             
    torch.manual_seed(2021)  # 设置主随机数生成器的种子
    torch.cuda.manual_seed(2021)  # 设置CUDA随机数生成器的种子
    random_rnn = MY_RNN(50, 50, random_embedding.len_words)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    random_cnn = MY_CNN(50, random_embedding.len_words, random_embedding.longest)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    glove_rnn = MY_RNN(50, 50, glove_embedding.len_words, weight=torch.tensor(glove_embedding.embedding, dtype=torch.float))
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    glove_cnn = MY_CNN(50, glove_embedding.len_words, glove_embedding.longest, weight=torch.tensor(glove_embedding.embedding, dtype=torch.float))
    # rnn+random
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    trl_ran_rnn, tel_ran_rnn, lol_ran_rnn, tra_ran_rnn, tes_ran_rnn, lon_ran_rnn = \
        NN_embdding(random_rnn, train_random, test_random, learning_rate, iter_times)
    # cnn+random
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    trl_ran_cnn,tel_ran_cnn,lol_ran_cnn, tra_ran_cnn, tes_ran_cnn, lon_ran_cnn = \
        NN_embdding(random_cnn, train_random, test_random, learning_rate, iter_times)
    # rnn+glove
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    trl_glo_rnn,tel_glo_rnn,lol_glo_rnn, tra_glo_rnn, tes_glo_rnn, lon_glo_rnn = \
        NN_embdding(glove_rnn, train_glove,test_glove, learning_rate, iter_times)
    # cnn+glove
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    trl_glo_cnn,tel_glo_cnn,lol_glo_cnn, tra_glo_cnn, tes_glo_cnn, lon_glo_cnn= \
        NN_embdding(glove_cnn,train_glove,test_glove, learning_rate, iter_times)
    
    save_folder = "./result/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
   	# 画图部分 
    x=list(range(1,iter_times+1))
    plt.subplot(2, 2, 1)
    plt.plot(x, trl_ran_rnn, 'r--', label='RNN+random')
    plt.plot(x, trl_ran_cnn, 'g--', label='CNN+random')
    plt.plot(x, trl_glo_rnn, 'b--', label='RNN+glove')
    plt.plot(x, trl_glo_cnn, 'y--', label='CNN+glove')
    plt.legend()
    plt.legend(fontsize=10)
    plt.title("Train Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.subplot(2, 2, 2)
    plt.plot(x, tel_ran_rnn, 'r--', label='RNN+random')
    plt.plot(x, tel_ran_cnn, 'g--', label='CNN+random')
    plt.plot(x, tel_glo_rnn, 'b--', label='RNN+glove')
    plt.plot(x, tel_glo_cnn, 'y--', label='CNN+glove')
    plt.legend()
    plt.legend(fontsize=10)
    plt.title("Test Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.subplot(2, 2, 3)
    plt.plot(x, tra_ran_rnn, 'r--', label='RNN+random')
    plt.plot(x, tra_ran_cnn, 'g--', label='CNN+random')
    plt.plot(x, tra_glo_rnn, 'b--', label='RNN+glove')
    plt.plot(x, tra_glo_cnn, 'y--', label='CNN+glove')
    plt.legend()
    plt.legend(fontsize=10)
    plt.title("Train Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.subplot(2, 2, 4)
    plt.plot(x, tes_ran_rnn, 'r--', label='RNN+random')
    plt.plot(x, tes_ran_cnn, 'g--', label='CNN+random')
    plt.plot(x, tes_glo_rnn, 'b--', label='RNN+glove')
    plt.plot(x, tes_glo_cnn, 'y--', label='CNN+glove')
    plt.legend()
    plt.legend(fontsize=10)
    plt.title("Test Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(8, 8, forward=True)
    plt.savefig(save_folder + 'main_plot.jpg')
    plt.clf()

    plt.subplot(2, 1, 1)
    plt.plot(x, lon_ran_rnn, 'r--', label='RNN+random')
    plt.plot(x, lon_ran_cnn, 'g--', label='CNN+random')
    plt.plot(x, lon_glo_rnn, 'b--', label='RNN+glove')
    plt.plot(x, lon_glo_cnn, 'y--', label='CNN+glove')
    plt.legend()
    plt.legend(fontsize=10)
    plt.title("Long Sentence Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.subplot(2, 1, 2)
    plt.plot(x, lol_ran_rnn, 'r--', label='RNN+random')
    plt.plot(x, lol_ran_cnn, 'g--', label='CNN+random')
    plt.plot(x, lol_glo_rnn, 'b--', label='RNN+glove')
    plt.plot(x, lol_glo_cnn, 'y--', label='CNN+glove')
    plt.legend()
    plt.legend(fontsize=10)
    plt.title("Long Sentence Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(8, 8, forward=True)
    plt.savefig(save_folder + 'sub_plot.jpg')
