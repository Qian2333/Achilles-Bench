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

The arguments for ``cal_rank.py``

|      |                  | description                                                  |
| ---- | ---------------- | ------------------------------------------------------------ |
| -s   | --dataset        | Choose the datasets you want to test with                    |
| -ts  | --num_split      | you can div the whole dataset into num_split and cal together for acceleration |
| -t   | --train          | train the model before calculate or not                      |
| -id  | --block_id       | the id of the block of datasets split into num_split sets    |
| -sd  | --seed           | choose the random seed                                       |
| -tr  | --attacked_model | the model used to test                                       |
| -md  | --mode           | gd for Hard-Bench(GradNorm), ls for Hard-Bench(Loss)         |

