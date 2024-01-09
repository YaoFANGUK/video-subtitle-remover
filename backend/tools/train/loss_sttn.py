import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    """
    对抗性损失
    根据论文 https://arxiv.org/abs/1711.10337 实现
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        可以选择的损失类型有 'nsgan' | 'lsgan' | 'hinge'
        type: 指定使用哪种类型的 GAN 损失。
        target_real_label: 真实图像的目标标签值。
        target_fake_label: 生成图像的目标标签值。
        """
        super(AdversarialLoss, self).__init__()
        self.type = type  # 损失类型
        # 使用缓冲区注册标签，这样在模型保存和加载时会一同保存和加载
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        # 根据选择的类型初始化不同的损失函数
        if type == 'nsgan':
            self.criterion = nn.BCELoss()  # 二进制交叉熵损失（非饱和GAN）
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()  # 均方误差损失（最小平方GAN）
        elif type == 'hinge':
            self.criterion = nn.ReLU()  # 适用于hinge损失的ReLU函数

    def __call__(self, outputs, is_real, is_disc=None):
        """
        调用函数计算损失。
        outputs: 网络输出。
        is_real: 如果是真实样本，则为 True；如果是生成样本，则为 False。
        is_disc: 指示当前是否在优化判别器。
        """
        if self.type == 'hinge':
            # 对于 hinge 损失
            if is_disc:
                # 如果是判别器
                if is_real:
                    outputs = -outputs  # 对真实样本反向标签
                # max(0, 1 - (真/假)示例输出)
                return self.criterion(1 + outputs).mean()
            else:
                # 如果是生成器, -min(0, -输出) = max(0, 输出)
                return (-outputs).mean()
        else:
            # 对于 nsgan 和 lsgan 损失
            labels = (self.real_label if is_real else self.fake_label).expand_as(
                outputs)
            # 计算模型输出和目标标签之间的损失
            loss = self.criterion(outputs, labels)
            return loss
