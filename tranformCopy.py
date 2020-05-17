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

def timeSince(since):
    "获得每次打印的训练耗时, since是训练开始时间"
    # 获得当前时间
    now = time.time()
    # 获得时间差，就是训练耗时
    s = now - since
    # 将秒转化为分钟, 并取整
    m = math.floor(s / 60)
    # 计算剩下不够凑成1分钟的秒数
    s -= m * 60
    # 返回指定格式的耗时
    return '%dm %ds' % (m, s)

# 该函数用于随机生成copy任务的数据
def data_generator(V, batch, num_batch):
    '''
    :param V: 随机生成数字的最大值+1
    :param batch: 每次输送给模型更新一次参数的数据量
    :param num_batch: 一共输送num_batch次完成一轮
    :return:
    '''

    # 使用for循环遍历nbatches
    for i in range(num_batch):
        # 在循环中使用np的random.randint方法随机生成[1, V)的整数,
        # 分布在(batch, 10)形状的矩阵中, 然后再把numpy形式转换称torch中的tensor.
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))

        # 接着使数据矩阵中的第一列数字都为1, 这一列也就成为了起始标志列,
        # 当解码器进行第一次解码的时候, 会使用起始标志列作为输入.
        data[:, 0] = 1

        # 因为是copy任务, 所有source与target是完全相同的, 且数据样本作用变量不需要求梯度
        # 因此requires_grad设置为False
        source = Variable(data, requires_grad=False).long()
        target = Variable(data, requires_grad=False).long()

        # 使用Batch对source和target进行对应批次的掩码张量生成, 最后使用yield返回
        yield Batch(source.to(device), target.to(device))

def run(model, loss, V, epochs=10):
    start = time.time()
    for epoch in range(epochs):
        model.train()

        # 每批次8个，一个批次循环20次
        run_epoch(data_generator(V, 8, 20), model, loss)

        model.eval()
        print("epochs:%d || time:%s " %(epoch+1, timeSince(start)))

        run_epoch(data_generator(V, 8, 5), model, loss)

    print("--------------------训练完毕，保存模型-----------------")
    torch.save(model.state_dict(), "./model/transform_"+TYPE+".pt")

    # 模型进入测试模式
    model.eval()

    # 假定的输入张量
    source = Variable(torch.LongTensor([[1,3,2,5,4,6,7,8,9,10]])).to(device)

    # 定义源数据掩码张量, 因为元素都是1, 在我们这里1代表不遮掩
    # 因此相当于对源数据没有任何遮掩.
    source_mask = Variable(torch.ones(1, 1, 10)).to(device)

    # 最后将model, src, src_mask, 解码的最大长度限制max_len, 默认为10
    # 以及起始标志数字, 默认为1, 我们这里使用的也是1
    result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print("输入的数据:", source)
    print("预测结果是:",result)

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
        print("输入数据为:",source)
        print("第{}数据预测的结果是:{}".format(num,result))
        num += 1

    print("准确率为:",acc_count/ len(sources))





if __name__ == '__main__':
    epochs = 40
    # 源数据特征(词汇)总数
    # 目标数据特征(词汇)总数
    source_vocab = target_vocab = 11
    # 编码器和解码器堆叠数
    N = 6
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

