# MP-VIB
Official implementation of IEEE Access paper "Multi-Prototype Variational Information Bottleneck for SSL-Based Speech Spoofing Detection"

## Datasets

**To test the generalization of the model, we selected multiple existing test sets, covering various types of spoofed speech attacks, including TTS, VC, Codec, Diffusion, Flow-matching, and others.**
- [ASVSpoof2019](https://zenodo.org/records/6906306)
- [ASVSpoof2021LA](https://zenodo.org/records/4837263)
- [ASVSpoof2021DF](https://zenodo.org/records/4837263)
- [ASVSpoof2024-Eval](https://zenodo.org/records/14498691)
- [FakeOrReal](https://bil.eecs.yorku.ca/share/for-norm.tar.gz)
- [Codecfake Yuankun et. al.](https://github.com/xieyuankun/Codecfake)
- [ADD 2022 Track 1](https://zenodo.org/records/10843991)
- [ADD 2022 Track 3](https://zenodo.org/records/12188055)
- [ADD 2023 R1](https://zenodo.org/records/12175884)
- [ADD 2023 R2](https://zenodo.org/records/12176326)
- [DFADD](https://github.com/isjwdu/DFADD)
- [LibriVoc](https://zenodo.org/records/15127251)
- [SONAR](https://github.com/Jessegator/SONAR)
- [In The Wild](https://deepfake-total.com/in_the_wild)

**Data Augment**
- [RIR](https://www.openslr.org/28/)
- [MUSAN](https://openslr.elda.org/17/)

## Installation
```
$ git clone https://github.com/Hench-Ho/MP-VIB.git
$ cd MP-VIB
$ conda env create -f environment.yml
```

## Pre-trained WavLM Model
Download the WavLM models from [here](https://github.com/microsoft/unilm/tree/master/wavlm) Place the downloaded weights under the wavlm folder.

## Inference
```
$ python main.py \
        --wavlm_path       wavlm/WavLM-Large.pt \
        --model_path       checkpoints/mp_vib_best.pth \
        --protocol_path    /path/to/eval_protocol.txt \
        --database_path    /path/to/eval_flac_dir \
        --output_path      scores.txt
```
