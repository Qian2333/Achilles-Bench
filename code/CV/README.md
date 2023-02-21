## Quick Start

you can run the code to test vgg16 with HardBench(Loss) on CIFAR-10 with 

```bash
python main_attack.py -s cifar10 -m tr -md vgg16 -tr c10_ls_ffn
```

you can also calculate the vgg16-based gradNrom score ofCIFAR-10 through

```bash
python get_rank.py -m vgg16
```

All the code default to use gpu. You need to check the code if you run the code on cpu.

you can change dataset by ```-s cifar100``` for CIFAR-100 and ```-s imagenet``` for ImageNet. 

Of course, you need to download ImageNet youself and set the path in the function ``DataIO.read_train_dev_test``  in  ``lib/dataset/get_data.py``. If the format you download is not compatible with ours, you may need to modify the data reading function yourself.

the table shows the models can be test with

| Model            | -md            |
| ---------------- | -------------- |
| FFN              | ffn            |
| VGG16            | vgg16          |
| ResNet18         | resnet18       |
| DenseNet121      | densenet121    |
| ViT-B/16         | ViT            |
| EfficientNetV2-S | EfficientNetV2 |

the list show the rank for different datasets calculated by different model

| -tr                  | description                                                  |
| -------------------- | ------------------------------------------------------------ |
| c10_ls_ffn_trained   | Hard-Bench (Loss) rank for CIFAR-10 calculated by trained FFN |
| c10_gd_ffn_trained   | Hard-Bench (GradNorm) rank for CIFAR-10 calculated by trained FFN |
| c100_ls_ffn_trained  | Hard-Bench (Loss) rank for CIFAR-100 calculated by trained FFN |
| c100_ffn_gdn_trained | Hard-Bench (GradNorm) rank for CIFAR-100 calculated by trained FFN |
| c10_resnet18_gd      | Hard-Bench (GradNorm) rank for CIFAR-10 calculated by trained ResNet18 |
| c10_ViT_gd           | Hard-Bench (GradNorm) rank for CIFAR-10 calculated by trained ViT-B/16 |
| img_ffn_trained_gdn  | Hard-Bench (GradNorm) rank for ImageNet calculated by trained FFN |
| img_ls_ffn_trained   | Hard-Bench (Loss) rank for ImageNet calculated by trained FFN |

You can calculate the rank with get_rank.py youself.

