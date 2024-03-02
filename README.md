# SCNet-Pytorch

Unofficial PyTorch implementation of the paper 
["SCNet: Sparse Compression Network for Music Source Separation"](https://arxiv.org/abs/2401.13276.pdf).

![architecture](images/architecture.png)

---
## Table of Contents

1. [Changelog & ToDo's](#changelog)
2. [Dependencies](#dependencies)
3. [Inference](#inference)
4. [Train](#train)
5. [Evaluate](#eval)
6. [Repository structure](#structure)
7. [Citing](#cite)

---
<a name="changelog"/>

# Changelog

- **10.02.2024**
  - Model itself is finished. The train script is on its way.
- **21.02.2024**
  - Add part of the training pipeline.
- **02.03.2024**
  - Finish the training pipeline and the separator.

# ToDo's:
- Add extensive README. 
- Add evaluation pipeline.
- Train some models.

---
<a name="cite"/>

# Citing

To cite this paper, please use:
```
@misc{tong2024scnet,
      title={SCNet: Sparse Compression Network for Music Source Separation}, 
      author={Weinan Tong and Jiaxu Zhu and Jun Chen and Shiyin Kang and Tao Jiang and Yang Li and Zhiyong Wu and Helen Meng},
      year={2024},
      eprint={2401.13276},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```



