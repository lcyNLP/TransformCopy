import TransForm.model as TransForm
import torch
import numpy as np
from torch.autograd import Variable
# 导入工具包Batch, 它能够对原始样本数据生成对应批次的掩码张量
from pyitcast.transformer_utils import Batch
# 导入优化器工具包get_std_opt, 该工具用于获得标准的针对Transformer模型的优化器
# 该标准优化器基于Adam优化器, 使其对序列到序列的任务更有效.
from pyitcast.transformer_utils import get_std_opt
# 因此使用标签平滑来弥补这种偏差, 减少模型对某一条规律的绝对认知, 以防止过拟合. 通过下面示例了解更多.
from pyitcast.transformer_utils import LabelSmoothing
import time
import math
import pandas as pd
# 导入损失计算工具包, 该工具能够使用标签平滑后的结果进行损失的计算,
# 损失的计算方法可以认为是交叉熵损失函数.
# from pyitcast.transformer_utils import SimpleLossCompute

# 导入贪婪解码工具包greedy_decode, 该工具将对最终结进行贪婪解码
# 贪婪解码的方式是每次预测都选择概率最大的结果作为输出,
# 它不一定能获得全局最优性, 但却拥有最高的执行效率.
from pyitcast.transformer_utils import greedy_decode
from pyitcast.transformer_utils import run_epoch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TYPE = "01"
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data.item() * norm
        # return loss.data[0] * norm

def predict(model, sources):
    # 模型进入测试模式
    # print(sources)
    checkpoint = torch.load("./model/transform_"+TYPE+".pt",map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    source_mask = Variable(torch.ones(1, 1, 10)).to(device)
    num = 1
    acc_count = 0
    for i in sources:
        source = Variable(torch.LongTensor([[int(j) for j in i.strip("\n").strip(" ").split(",")]])).to(device)
        result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)

        if torch.equal(source, result):
            acc_count += 1
        print("第{}     输入数据为:{}".format(num,source))
        print("第{}数据预测的结果是:{}".format(num,result))
        num += 1

    print("准确率为:",acc_count/ len(sources))





if __name__ == '__main__':
    epochs = 50
    # 源数据特征(词汇)总数
    # 目标数据特征(词汇)总数
    source_vocab = target_vocab = 11
    # 编码器和解码器堆叠数
    N = 2
    # 词向量映射维度
    d_model = 512
    # 前馈全连接网络中变换矩阵的维度
    d_ff = 2048
    # 多头注意力结构中的多头数
    head = 8
    # 置零比率
    dropout = 0.1

    # 初始化模型
    model = TransForm.make_model(source_vocab, target_vocab,
                                 N=N, d_model=d_model,
                                 d_ff=d_ff, head=head,
                                 dropout=dropout).to(device)
    # 使用get_std_opt获得模型优化器
    model_optimizer = get_std_opt(model)
    # 使用LabelSmoothing获得标签平滑对象
    criterion = LabelSmoothing(size=source_vocab, padding_idx=0, smoothing=0.0)
    # 使用SimpleLossCompute获得利用标签平滑结果的损失计算方法
    loss = SimpleLossCompute(model.generator, criterion, model_optimizer)

    # 模型训练过程
    # run(model, loss, source_vocab, epochs)

    # 模型预测过程
    data = open("./data/test_100.csv").readlines()
    predict(model, data)

