import torchvision.models as models
import torch.nn as nn
import timm
import torch

def vgg(train=False):
    model = models.vgg11(num_classes=10)
    if train:
        model.train()
        model.avgpool = nn.MaxPool2d(1)
        model.classifier[2] = nn.Identity()
        model.classifier[5] = nn.Identity()
    else:
        model.eval()
        model.avgpool = nn.AvgPool2d(1)
    return model

def alexnet(train=False):
    model = models.alexnet(num_classes=10)
    if train:
        model.train()
        model.avgpool = nn.MaxPool2d(1)
        model.classifier[0] = nn.Identity()
        model.classifier[3] = nn.Identity()
    else:
        model.eval()
        model.avgpool = nn.AvgPool2d(1)
    return model

def mobilenet_v2(train=False):
    model = models.mobilenet_v2(num_classes=10)
    if train:
        model.train()
        model.classifier[0] = nn.Identity()
    else:
        model.eval()
    return model

def efficientnet_b0(train=False):
    model = models.efficientnet_b0(num_classes=10)
    if train:
        model.train()
        model.avgpool = nn.MaxPool2d(kernel_size=7)
        model.classifier[0] = nn.Identity()
    else:
        model.eval()
        model.avgpool = nn.AvgPool2d(kernel_size=7)
    return model

def resnet18(train=False):
    model = models.resnet18(num_classes=10)
    if train:
        model.train()
        model.avgpool = nn.MaxPool2d(kernel_size=7)
    else:
        model.eval()
        model.avgpool = nn.AvgPool2d(kernel_size=7)
    return model

def vit_b_16(train=False):
    model = models.vit_b_16(num_classes=10)
    if train:
        model.train()
    else:
        model.eval()
    return model

def vit_tiny_patch16_224(train=False):
    model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
    if train:
        model.train()
    else:
        model.eval()
    return model

def mobilevit_s(train=False):
    model = timm.create_model("mobilevit_s", pretrained=True)
    if train:
        model.train()
    else:
        model.eval()
    return model

model_dict = {
    "vgg": vgg,
    "alexnet": alexnet,
    "mobilenet_v2": mobilenet_v2,
    "efficientnet_b0": efficientnet_b0,
    "resnet18": resnet18,
    "vit_b_16": vit_b_16,
    "vit_tiny_patch16_224": vit_tiny_patch16_224,
    "mobilevit_s": mobilevit_s,
}

