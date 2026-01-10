# models/resnet_encoder.py
'''
ResNet-50 属于 Feature Extractor 的“场景外观编码器”，负责把当前第一视角 RGB 场景压成一个固定长度向量 
    输入：当前帧 RGB（400×300）
    输出：2048 维视觉特征 
    关键点：预训练 ResNet-50 + 参数冻结
这个视觉特征会被存进 OSM（Object-Scene Memory）
'''

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

class ResNet50Encoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # 预训练 ResNet-50
        self.backbone = nn.Sequential(*list(net.children())[:-1])  # 去掉最后的分类层, 输出2048维特征
        for p in self.backbone.parameters():
            p.requires_grad = False # 冻结参数
        self.backbone.eval() # 设置为评估模式
        self.device = device
        self.to(device)
        self.tf = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(mean=models.ResNet50_Weights.DEFAULT.transforms().mean,
                        std=models.ResNet50_Weights.DEFAULT.transforms().std),
        ])

    @torch.no_grad()
    def encode(self, rgb_uint8):  # HxWx3 uint8
        x = self.tf(rgb_uint8).unsqueeze(0).to(self.device)  # (1,3,224,224)
        feat = self.backbone(x).flatten(1)  # (1,2048)
        return feat.squeeze(0).cpu()
