# 实验设置
* 特征表示
    * random embedding
    * glove embedding
* 模型
    * Bi-LSTM + CRF 
* 损失
    * loss: -(True_score(X,y)−log(Total_score))
* 网络参数
    * len_feature = 50
    * len_hidden = 50
    * hidden_layer=1
* 训练参数
    * learning_rate = 0.001
    * iter_times: 100
    * batch_size: 100
    * drop_out: 0.5
# 实验结果
## 训练和测试
![train and test](./main_plot.jpg)
# 结果分析