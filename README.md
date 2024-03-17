# SCNet-Pytorch

Unofficial PyTorch implementation of the paper 
["SCNet: Sparse Compression Network for Music Source Separation"](https://arxiv.org/abs/2401.13276.pdf).

![architecture](images/architecture.png)

---
## Table of Contents

1. [Changelog & ToDo's](#changelog)
2. [Dependencies](#dependencies)
3. [Training](#train)
4. [Inference](#inference)
5. [Evaluation](#eval)
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
- **17.03.2024**
  - Finish inference.py and fill README.md

# ToDo's:
- Add evaluation pipeline.

---
<a name="dependencies"/>

# Dependencies

Before starting training, you need to install the requirements:

```bash
pip install -r requirements.txt
```

Then, download the MUSDB18HQ dataset:

```bash
wget -P /path/to/dataset/musdb18hq.zip https://zenodo.org/records/3338373/files/musdb18hq.zip 
unzip /path/to/dataset/musdb18hq.zip -d /path/to/dataset
```

Next, create environment variables with paths to the audio data and generated metadata `.pqt` file:

```bash
export DATASET_DIR=/path/to/dataset/musdb18hq
export DATASET_PATH=/path/to/dataset/dataset.pqt
```

Finally, export the GPU to make it visible:
```bash
export CUDA_VISIBLE_DEVICES=0
```

Now, you can train the model.

---
<a name="train"/>

# Training

To train the model, a combination of `PyTorch-Lightning` and `hydra` was used.
All configuration files are stored in the `src/conf` directory in `hydra`-friendly format.

To start training a model with given configurations, use the following script:
```
python src/train.py
```
To configure the training process, follow `hydra` [instructions](https://hydra.cc/docs/advanced/override_grammar/basic/).
You can modify/override the arguments doing something like this:
```
python src/train.py +trainer.overfit_batches=10 loader.train.batch_size=16
```

After training is started, the logging folder will be created for a particular experiment with the following path:
```
logs/scnet/${now:%Y-%m-%d}_${now:%H-%M}/
```
This folder will have the following structure:
```
├── checkpoints
│   └── tensorboard_log_file    - main tensorboard log file 
├── tensorboard
│   └── *.ckpt                  - lightning model checkpoint files.
└── yamls
│   └──*.yaml                   - hydra configuration and override files 
└── train.log                   - logging file for train.py
   
```

---
<a name="inference"/>

# Inference

After training a model, you can run inference using the following command:

```bash
python src/inference.py -i <INPUT_PATH> \ 
                        -o <OUTPUT_DIR> \
                        -c <CHECKPOINT_PATH> \ 
```

This command will generate separated audio files in .wav format in the <OUTPUT_DIR> directory.

For more information about the script and its options, use:
```bash
usage: inference.py [-h] -i INPUT_PATH -o OUTPUT_PATH -c CKPT_PATH [-d DEVICE] [-b BATCH_SIZE] [-w WINDOW_SIZE] [-s STEP_SIZE] [-p]

Argument Parser for Separator

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input-path INPUT_PATH
                        Input path to .wav audio file/directory containing audio files
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        Output directory to save separated audio files in .wav format
  -c CKPT_PATH, --ckpt-path CKPT_PATH
                        Path to the model checkpoint
  -d DEVICE, --device DEVICE
                        Device to run the model on (default: cuda)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size for processing (default: 4)
  -w WINDOW_SIZE, --window-size WINDOW_SIZE
                        Window size (default: 11)
  -s STEP_SIZE, --step-size STEP_SIZE
                        Step size (default: 5.5)
  -p, --use-progress-bar
                        Use progress bar (default: True)
```

Additionally, you can run inference within Python using the following script:
```python
import sys
sys.path.append('src/')

import torchaudio
from src.model.separator import Separator

device: str = 'cuda'

separator = Separator.load_from_checkpoint(
    path="<CHECKPOINT_PATH>",   # path to trained Lightning checkpoint
    batch_size=4,         # adjust batch size to fit into your GPU's memory
    window_size=11,       # window size of the model (do not change)
    step_size=5.5,        # as step size is closer to window size, inference will be faster, but results less good
    use_progress_bar=True # show progress bar per audio file
).to(device)

y, sr = torchaudio.load("<INPUT_PATH>")
y = y.to(device)

y_separated = separator.separate(y).cpu()
```

Make sure to replace `<INPUT_PATH>`, `<OUTPUT_DIR>`, and `<CHECKPOINT_PATH>` with the appropriate paths for your setup.

---
<a name="eval"/>

# Evaluation

...

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



