## Quick Start

you can run the code to test t5 with HardBench(Loss) on sst2 with 

```bash
python main_attack.py -s sst2 -m tr -md t5 -tr sst2_ls_bert_t
```

you can also calculate the roberta-based gradNrom score of sst2 through

```bash
python cal_rank.py -tr roberta
```

All the code default to use gpu. You need to check the code if you run the code on cpu.

you can change dataset by ``-s mnli`` for MNLI and so on (In lower case). 

Of course, you need to download ImageNet youself and set the path in the function ``DataIO.read_train_dev_test``  in  ``lib/dataset/get_data.py``. If the format you download is not compatible with ours, you may need to modify the data reading function yourself.

the table shows the models can be test with

| Model       | -md               |
| ----------- | ----------------- |
| bert        | Bert-base-uncased |
| gpt2        | GPT2              |
| roberta     | RoBerta-base      |
| t5          | T5-base           |
| transformer | transformer       |

the list show the rank for different datasets calculated by different model

| -tr                   | description                                                  |
| --------------------- | ------------------------------------------------------------ |
| sst2_bert_gd_t        | Hard-Bench (GradNorm) rank for SST2 calculated by trained BERT |
| sst2_transformer_gd_t | Hard-Bench (GradNorm) rank for SST2 calculated by trained Transformer |
| sst2_gpt_gd_t         | Hard-Bench (GradNorm) rank for SST2 calculated by trained GPT2 |
| cola_bert_gd_t        | Hard-Bench (GradNorm) rank for COLA calculated by trained BERT |
| mnli_bert_gd_t        | Hard-Bench (GradNorm) rank for MNLI calculated by trained BERT |
| mrpc_bert_gd_t        | Hard-Bench (GradNorm) rank for MRPC calculated by trained BERT |
| rte_bert_gd_t         | Hard-Bench (GradNorm) rank for RTE calculated by trained BERT |
| wnli_bert_gd_t        | Hard-Bench (GradNorm) rank for WNLI calculated by trained BERT |
| qnli_bert_gd_t        | Hard-Bench (GradNorm) rank for QNLI calculated by trained BERT |
| qqp_bert_gd_t         | Hard-Bench (GradNorm) rank for QQP calculated by trained BERT |
| stsb_bert_gd_t        |  Hard-Bench (GradNorm) rank for STSB calculated by trained BERT  |
| sst2_ls_bert_t | Hard-Bench (Loss) rank for SST2 calculated by trained BERT |
| cola_ls_bert_t| Hard-Bench (Loss) rank for COLA calculated by trained BERT |
| mnli_ls_bert_t| Hard-Bench (Loss) rank for MNLI  calculated by trained BERT |
| wnli_ls_bert_t| Hard-Bench (Loss) rank for WNLI  calculated by trained BERT |
| qqp_ls_bert_t| Hard-Bench (Loss) rank for QQP  calculated by trained BERT |
| rte_ls_bert_t| Hard-Bench (Loss) rank for RTE  calculated by trained BERT |
| mrpc_ls_bert_t| Hard-Bench (Loss) rank for MRPC  calculated by trained BERT |
| qnli_ls_bert_t| Hard-Bench (Loss) rank for QNLI  calculated by trained BERT |

You can calculate the rank with cal_rank.py youself.

