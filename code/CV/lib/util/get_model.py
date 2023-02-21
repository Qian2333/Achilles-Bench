from lib.model.cifarnet import Net as ffn
from lib.model.cifarnet import ImgNet as ffn1000
from lib.model.vgg import VGG
from lib.model.resnet import ResNet18
from lib.model.densenet import DenseNet121
from lib.model.vit import Vit
from lib.model.effv2 import Effnet
import torchvision.models as models
from robustbench.utils import load_model
from robustbench.data import load_cifar10


robust_bench = ['Rebuffi2021Fixing_70_16_cutmix_extra', 'Gowal2021Improving_70_16_ddpm_100m',
                'Rade2021Helper_extra', 'Rebuffi2021Fixing_70_16_cutmix_ddpm',
                'Gowal2020Uncovering_extra', 'Debenedetti2022Light_XCiT-L12']


def get_cifar10_model(model_name):
    net = ffn(num_cls=10)
    if model_name == 'vgg16':
        net = VGG('VGG16', num_cls=10)
    elif model_name == 'resnet18':
        net = ResNet18(num_cls=10)
    elif model_name == 'densenet121':
        net = DenseNet121(num_cls=10)
    elif model_name in robust_bench:
        net = load_model(model_name=model_name,
                         dataset='cifar10', threat_model='corruptions')
    elif model_name == 'ViT':
        net = Vit(num_cls=10)
    elif model_name == 'EfficientNetV2':
        net = Effnet(num_cls=10)
    return net


def get_cifar100_model(model_name):
    net = ffn(num_cls=100)
    if model_name == 'vgg16':
        net = VGG('VGG16', num_cls=100)
    elif model_name == 'resnet18':
        net = ResNet18(num_cls=100)
    elif model_name == 'densenet121':
        net = DenseNet121(num_cls=100)
    elif model_name in robust_bench:
        net = load_model(model_name=model_name,
                         dataset='cifar100', threat_model='corruptions')
    elif model_name == 'ViT':
        net = Vit(num_cls=100)
    elif model_name == 'EfficientNetV2':
        net = Effnet(num_cls=100)
    return net


def get_imagenet_model(model_name):
    net = ffn1000(num_cls=1000)
    if model_name == 'vgg16':
        net = models.vgg16()
    elif model_name == 'resnet18':
        net = models.resnet18()
    elif model_name == 'densenet121':
        net = models.densenet121()
    elif model_name in robust_bench:
        net = load_model(model_name=model_name,
                         dataset='imagenet', threat_model='corruptions')
    elif model_name == 'ViT':
        net = Vit(num_cls=1000)
    elif model_name == 'EfficientNetV2':
        net = Effnet(num_cls=1000)
    return net


def get_model(model_name, dataset):
    if dataset == 'cifar10' or dataset == 'mnist':
        return get_cifar10_model(model_name)
    elif dataset == 'cifar100':
        return get_cifar100_model(model_name)
    elif dataset == 'imagenet':
        return get_imagenet_model(model_name)




