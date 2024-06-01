# Achilles-Bench: A Challenging Benchmark for Low-Resource Evaluation

Paper: [[2303.03840\] A Challenging Benchmark for Low-Resource Learning (arxiv.org)](https://arxiv.org/abs/2303.03840)

LeaderBoard:[Achilles-Bench (qian2333.github.io)](https://qian2333.github.io/Hard-Bench-Web/)

the samulation code:https://colab.research.google.com/drive/1pywuN8W741kOEEDGqUJXf_dCKhLqar7d?usp=sharing

### Main idea:

With promising yet saturated results in high-resource settings, low-resource datasets have gradually become popular benchmarks for evaluating the learning ability of advanced neural networks (e.g., BigBench, superGLUE). Some models even surpass humans according to benchmark test results. 

#### Achilles-Bench (Loss)

For each label, we choose top-k hard examples based on losses scores.

#### Achilles-Bench (GradNorm)

For each label, we choose top-k hard examples based on gradient norm scores.

#### Code

The code can be view in the folder.

While the data can be download in https://drive.google.com/drive/folders/12ThBP3NocuCgehskljItrwXVyk_EfwED?usp=share_link. （As the test data for ImageNet take up too much space, we only included train data for ImageNet）

#### Citation

Consider citing our paper:

```latex
 @misc{https://doi.org/10.48550/arxiv.2303.03840,
 doi = {10.48550/ARXIV.2303.03840},
 url = {https://arxiv.org/abs/2303.03840},
 author = {Wang, Yudong and Ma, Chang and Dong, Qingxiu and Kong, Lingpeng and Xu, Jingjing},
 keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
 title = {A Challenging Benchmark for Low-Resource Learning},
 publisher = {arXiv},
 year = {2023},
 copyright = {Creative Commons Attribution 4.0 International}
}

```

