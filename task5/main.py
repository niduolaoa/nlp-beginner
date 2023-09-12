import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from models import LSTM
from utils import data_loader
import math
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.manual_seed(2021)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
batch_size = 128
hidden_size = 512
num_layers = 2
epochs = 500
drop_out = 0.2
learning_rate = 0.01
MOMENTUM = 0.9
CLIP = 5
decay_rate = 0.05  # learning rate decay rate
EOS_TOKEN = "[EOS]"
data_file_path = '../../datasets/poetryFromTang.txt'
embedding_size = 300
temperature = 0.8  # Higher temperature means more diversity.
max_length = 128


def train(train_iter, dev_iter, loss_func, optimizer, epochs, clip):
    train_perplexity = []
    dev_perplexity = []
    for epoch in trange(epochs):
        model.train()
        total_loss = 0
        total_words = 0
        for i, batch in enumerate(train_iter):
            text, lens = batch.text
            inputs = text[:, :-1]
            targets = text[:, 1:]
            init_hidden = model.init_hidden(inputs.size(0))
            logits = model(inputs, lens - 1, init_hidden)  # [EOS] is included in length.
            loss = loss_func(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += loss.item()
            total_words += lens.sum().item()
        if epoch % 10 == 9:
            tqdm.write("Epoch: %d, Train perplexity: %d" % (epoch + 1, math.exp(total_loss / total_words)))
            dev_perplexity += eval(dev_iter, True, epoch)
        train_perplexity.append(math.exp(total_loss / total_words))

        lr = learning_rate / (1 + decay_rate * (epoch + 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return train_perplexity, dev_perplexity

def eval(data_iter, is_dev=False, epoch=None):
    model.eval()
    perplexity = []
    with torch.no_grad():
        total_words = 0
        total_loss = 0
        for i, batch in enumerate(data_iter):
            text, lens = batch.text
            inputs = text[:, :-1]
            targets = text[:, 1:]
            model.zero_grad()
            init_hidden = model.init_hidden(inputs.size(0))
            logits = model(inputs, lens - 1, init_hidden)  # [EOS] is included in length.
            loss = loss_func(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            total_loss += loss.item()
            total_words += lens.sum().item()
    if epoch is not None:
        tqdm.write(
            "Epoch: %d, %s perplexity %.3f" % (
                epoch + 1, "Dev" if is_dev else "Test", math.exp(total_loss / total_words)))
        perplexity.append(math.exp(total_loss / total_words))
    else:
        tqdm.write(
            "%s perplexity %.3f" % ("Dev" if is_dev else "Test", math.exp(total_loss / total_words)))
        perplexity.append(math.exp(total_loss / total_words))
    return perplexity

def generate(eos_idx, word, temperature=0.8):
    model.eval()
    with torch.no_grad():
        if word in TEXT.vocab.stoi:
            idx = TEXT.vocab.stoi[word]
            inputs = torch.tensor([idx])
        else:
            print("%s is not in vocabulary, choose by random." % word)
            prob = torch.ones(len(TEXT.vocab.stoi))
            inputs = torch.multinomial(prob, 1)
            idx = inputs[0].item()

        inputs = inputs.unsqueeze(1).to(device)
        lens = torch.tensor([1]).to(device)
        hidden = tuple([h.to(device) for h in model.lstm.init_hidden(1)])
        poetry = [TEXT.vocab.itos[idx]]

        while idx != eos_idx:
            logits, hidden = model(inputs, lens, hidden)
            word_weights = logits.squeeze().div(temperature).exp().cpu()
            idx = torch.multinomial(word_weights, 1)[0].item()
            inputs.fill_(idx)
            poetry.append(TEXT.vocab.itos[idx])
        print("".join(poetry[:-1]))


if __name__ == "__main__":
    train_iter, dev_iter, test_iter, TEXT = data_loader(EOS_TOKEN, batch_size, device, data_file_path, max_length)
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    eos_idx = TEXT.vocab.stoi[EOS_TOKEN]
    model = LSTM(len(TEXT.vocab), embed_size=embedding_size, hidden_size=hidden_size,
                 dropout_rate=drop_out, layer_num=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
    # 训练集和验证集上面的困惑度
    train_perplexity, dev_perplexity = train(train_iter, dev_iter, loss_func, optimizer, epochs, CLIP)
    # 测试集上面的困惑度
    test_perplexity = eval(test_iter, is_dev=False)
    # 作图
    save_folder = "./result/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(train_perplexity)), np.array(train_perplexity))
    plt.xlabel('Iterations')
    plt.ylabel('Perplexity')
    plt.title('Train Perplexity')
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(dev_perplexity)), np.array(dev_perplexity))
    plt.xlabel('Iterations')
    plt.ylabel('Perplexity')
    plt.title('Dev Perplexity')
    plt.text(0.5, 0.5, f"Test Perplexity: {test_perplexity[0]}", ha="center", fontsize=12)
    plt.subplots_adjust(wspace=0.5, hspace=0.2)
    plt.savefig(save_folder + "perplexity.jpg")
    # try:
    #     while True:
    #         word = input("Input the first word or press Ctrl-C to exit: ")
    #         generate(eos_idx, word.strip(), temperature)
    # except:
    #     pass

