import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self,layer_num:int,layer_links:list[int]):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            #[1,10],[10,100],[100,100],[100,10],[10,1]
            self.layers.append(nn.Linear(layer_links[i], layer_links[i + 1]))

    def forward(self, x):
        # 遍历所有层并应用ReLU激活函数，除了最后一层
        for layer in self.layers[:-1]:
            x = F.silu(layer(x))

        # 最后一层不使用激活函数
        x = self.layers[-1](x)
        return x

def create_model(layer_num,layer_links):
    # 创建MLP模型
    model = MLP(layer_num, layer_links)
    return model
# 示例用法
if __name__ == "__main__":
    # 定义层数和每层的神经元数量
    layer_num = 5
    layer_links = [1, 10, 100, 100, 10, 1]  # 输入层10个神经元，第一隐藏层20个神经元，第二隐藏层30个神经元，输出层10个神经元

    # 创建MLP模型
    model = MLP(layer_num, layer_links)

    # 打印模型结构
    print(model)

    # 创建一个示例输入张量
    x = torch.randn(5, 1)  # batch size为5，输入特征为10

    # 前向传播
    output = model(x)
    print(output)