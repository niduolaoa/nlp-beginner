import matplotlib.pyplot as plt
import torch
from feature import get_batch
from torch import optim
from neural_network import Named_Entity_Recognition
import os


def NN_embdding(model, train, test, learning_rate, iter_times,batch_size):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_record = list()
    test_loss_record = list()
    train_record = list()
    test_record = list()
    # torch.autograd.set_detect_anomaly(True)

    for iteration in range(iter_times):
        model.train()
        for i, batch in enumerate(train):
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            mask = (y != 0).cuda()
            loss = model(x, y, mask).cuda()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        train_acc = list()
        test_acc = list()
        train_loss = 0
        test_loss = 0
        for i, batch in enumerate(train):
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            mask = (y != 0).cuda()
            loss = model(x, y, mask).cuda()
            train_loss += loss.item() / batch_size / y.shape[1]
            pred = model.predict(x, mask)
            acc = (pred == y).float()
            len_batch,len_seq=acc.shape
            points=torch.ones((1,len_batch)).cuda()
            for j in range(len_seq):
              points*=acc[:,j]
            train_acc.append(points.mean())

        for i, batch in enumerate(test):
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            mask = (y != 0).cuda()
            loss = model(x, y, mask).cuda()
            test_loss += loss.item() / batch_size / y.shape[1]
            pred = model.predict(x, mask)
            acc = (pred == y).float()
            len_batch,len_seq=acc.shape
            points=torch.ones((1,len_batch)).cuda()
            for j in range(len_seq):
              points*=acc[:,j]
            test_acc.append(points.mean())

        trains_acc = sum(train_acc) / len(train_acc)
        tests_acc = sum(test_acc) / len(test_acc)

        train_loss_record.append(train_loss / len(train))
        test_loss_record.append(test_loss / len(test))
        train_record.append(trains_acc.cpu())
        test_record.append(tests_acc.cpu())
        print("---------- Iteration", iteration + 1, "----------")
        print("Train loss:", train_loss / len(train))
        print("Test loss:", test_loss / len(test))
        print("Train accuracy:", trains_acc)
        print("Test accuracy:", tests_acc)

    return train_loss_record, test_loss_record, train_record, test_record


def NN_plot(random_embedding, glove_embedding, len_feature, len_hidden, learning_rate, batch_size, iter_times):
    train_random = get_batch(random_embedding.train_x_matrix, random_embedding.train_y_matrix, batch_size)
    test_random = get_batch(random_embedding.test_x_matrix, random_embedding.test_y_matrix, batch_size)
    train_glove = get_batch(glove_embedding.train_x_matrix, glove_embedding.train_y_matrix, batch_size)
    test_glove = get_batch(glove_embedding.test_x_matrix, glove_embedding.test_y_matrix, batch_size)
    random_model = Named_Entity_Recognition(len_feature, random_embedding.len_words, len_hidden,
                                            random_embedding.len_tag, 0, 1, 2)
    glove_model = Named_Entity_Recognition(len_feature, random_embedding.len_words, len_hidden,
                                           random_embedding.len_tag, 0, 1, 2,
                                           weight=torch.tensor(glove_embedding.embedding, dtype=torch.float))
    trl_ran, tsl_ran, tra_ran, tea_ran = NN_embdding(random_model, train_random, test_random, learning_rate,
                                                     iter_times,batch_size)
    trl_glo, tsl_glo, tra_glo, tea_glo = NN_embdding(glove_model, train_glove, test_glove, learning_rate,
                                                     iter_times,batch_size)
    x = list(range(1, iter_times + 1))

    # 作图
    save_folder = "./result/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.subplot(2, 2, 1)
    plt.plot(x, trl_ran, 'r--', label='random')
    plt.plot(x, trl_glo, 'g--', label='glove')
    plt.legend(fontsize=10)
    plt.title("Train Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.subplot(2, 2, 2)
    plt.plot(x, tsl_ran, 'r--', label='random')
    plt.plot(x, tsl_glo, 'g--', label='glove')
    plt.legend(fontsize=10)
    plt.title("Test Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.subplot(2, 2, 3)
    plt.plot(x, tra_ran, 'r--', label='random')
    plt.plot(x, tra_glo, 'g--', label='glove')
    plt.legend(fontsize=10)
    plt.title("Train Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.subplot(2, 2, 4)
    plt.plot(x, tea_ran, 'r--', label='random')
    plt.plot(x, tea_glo, 'g--', label='glove')
    plt.legend(fontsize=10)
    plt.title("Test Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(8, 8, forward=True)
    plt.savefig(save_folder + 'main_plot.jpg')