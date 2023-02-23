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

The arguments for ``main_attack.py``

|      |                   | description                                                  |
| ---- | ----------------- | ------------------------------------------------------------ |
| -s   | --dataset         | Choose the datasets you want to test with                    |
| -m   | --mode            | mode of Random-Bench or Hard-Bench                           |
| -b   |                   | set the k for selecting k-shot sets                          |
| -r   | --test_round      | the round of test                                            |
| -rev | --reverse         | choose the highest rank data point or lowest(while rev=1 for hightest, 0 for lowest) |
| -md  | --attacked_model  | the model used to test                                       |
| -tr  | --attacking_model | the rank list you want to choose                             |
| -d   | --device          | use gpu or cpu                                               |
| -ts  | --training_set    | you can use data augment: cutmix or mixup                    |
| -sd  | --seed_begin      | the rounds will be seeded with seed_begin+[0, 1, 2, ..., test_round - 1] |

The arguments for ``get_rank.py``

|      |                  | description                                                  |
| ---- | ---------------- | ------------------------------------------------------------ |
| -s   | --dataset        | Choose the datasets you want to test with                    |
| -ts  | --num_split      | you can div the whole dataset into num_split and cal together for acceleration |
| -t   | --train          | train the model before calculate or not                      |
| -id  | --block_id       | the id of the block of datasets split into num_split sets    |
| -sd  | --seed           | choose the random seed                                       |
| -tr  | --attacked_model | the model used to test                                       |
| -en  | --epoch_num      | epoch for training                                           |
| -d   | --device         | gpu or cpu                                                   |
| -md  | --mode           | gd for Hard-Bench(GradNorm), ls for Hard-Bench(Loss)         |

