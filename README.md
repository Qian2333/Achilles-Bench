# A Challenging Benchmark for Low-Resource Learning

Paper: 

LeaderBoard:[Hard-Bench (qian2333.github.io)](https://qian2333.github.io/Hard-Bench-Web/)

the samulation code:https://colab.research.google.com/drive/1pywuN8W741kOEEDGqUJXf_dCKhLqar7d?usp=sharing

### Main idea:

With promising yet saturated results in high-resource settings, low-resource datasets have gradually become popular benchmarks for evaluating the learning ability of advanced neural networks (e.g., BigBench, superGLUE). Some models even surpass humans according to benchmark test results. 

#### Hard-Bench (Loss)

For each label, we choose top-k hard examples based on losses scores.

#### Hard-Bench (GradNorm)

For each label, we choose top-k hard examples based on gradient norm scores.

#### Code

The code can be view in the folder.

While the data can be download in https://drive.google.com/drive/folders/12ThBP3NocuCgehskljItrwXVyk_EfwED?usp=share_link. （As the test data for ImageNet take up too much space, we only included train data for ImageNet）

#### Citation

Consider citing our paper:

```latex
@inproceedings{???,
  title     = {???},
  author    = {???},
  booktitle = {???},
  year      = {???},
  url       = {???}
}
```

