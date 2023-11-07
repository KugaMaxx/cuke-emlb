# E-MLB: Multilevel Benchmark for Event-Based Camera Denoising

![](https://img.shields.io/github/v/tag/KugaMaxx/cuke-emlb?style=flat-square)
![](https://img.shields.io/github/license/KugaMaxx/cuke-emlb?style=flat-square)

The simple benchmark for event-based denoising. Any questions please contact me with [KugaMaxx@outlook.com](mailto:KugaMaxx@outlook.com).

<span id="animation"></span>
![animation](https://raw.githubusercontent.com/KugaMaxx/cuke-emlb/main/assets/images/animation.gif "animation")

## Installation

### Dependencies

To ensure the running of the project, the following dependencies are need.

+ Install common dependencies.

```bash
# Install compiler
sudo apt-get install git gcc-10 g++-10 cmake

# Install boost, opencv, eigen3, openblas
sudo apt-get install libboost-dev, libopencv-dev, libeigen3-dev, libopenblas-dev
```

+ Install third-party dependencies for dv.

```bash
# Add repository
sudo add-apt-repository ppa:inivation-ppa/inivation

# Update
sudo apt-get update

# Install pre dependencies
sudo apt-get install boost-inivation libcaer-dev libfmt-dev liblz4-dev libzstd-dev libssl-dev

# Install dv
sudo apt-get install dv-processing
```

+ Initialize our [dv-toolkit](https://github.com/KugaMaxx/yam-toolkit), for
simplifying the processing of event-based data.

```bash
# Recursively initialize our submodule
git submodule update --init --recursive
```

## Build from source

### For DV software usage

By following the steps below, you will obtain a series of `.so` files in the 
`./modules` folder, which are third-party modules that can be called by 
[DV software](https://inivation.gitlab.io/dv/dv-docs/docs/getting-started/). 
For how to use them, please refer to the "set up for dv" in the 
[tutorial](https://inivation.gitlab.io/dv/dv-docs/docs/first-module/).

+ Install dependencies for building modules.

```bash
sudo apt-get install dv-runtime-dev
```

+ Compile with setting `-DEMLB_ENABLE_MODULES`

```bash
# create folder
mkdir build && cd build

# compile with samples
CC=gcc-10 CXX=g++-10 cmake .. -DEMLB_ENABLE_MODULES=ON

# generate library
cmake --build . --config Release
```

### For script usage

In this section, the model will be built as Python packages by using pybind11, 
allowing directly import in your project. If using C++ language, you can 
directly copy the header files in `./include` and following 
[tutorial](https://dv-processing.inivation.com/rel_1.7/event_filtering.html) to
see how to use.

+ Install dependencies for building packages.

```bash
sudo apt-get install python3-dev python3-pybind11
```

+ Create a new virtual environment:

```bash
# Create virtual environment
conda create -n emlb python=3.8

# Activate virtual environment
conda activate emlb

# Install requirements
pip install -r requirements.txt
```

+ Compile with setting `-DEMLB_ENABLE_PYTHON`

```bash
# create folder
mkdir build && cd build

# compile with samples
CC=gcc-10 CXX=g++-10 cmake .. -DEMLB_ENABLE_PYTHON=ON

# generate library
cmake --build . --config Release
```
+ Run `demo.py` to test:

```bash
python3 demo.py
```

### CUDA support <span id="cuda"></span>

Assuming that [libtorch](https://pytorch.org/cppdocs/installing.html) is 
installed, you can include `-DTORCH_DIR=/path/to/libtorch/` to compile deep 
learning models. For example, you can build by following instruction.

```bash
CC=gcc-10 CXX=g++-10 cmake .. \
-DEMLB_ENABLE_PYTHON=ON \
-DTORCH_DIR=<path/to/libtorch>/share/cmake/Torch/
```

NOTE: download pretrained models [here](https://drive.google.com/drive/folders/1BytQnsNRlv1rJyMotElIqklOz1oCt2Vd?usp=sharing) 
and paste them into `./modules/net/` folder.

## Inference with SOTA

At present, we have implemented the following event-based denoising algorithms. 

| Algorithms        | Full Name                     | Year | Languages | DV  | Cuda |
| :---------------: | :---------------------------: | :--: | :-------: | :-: | :--: |
| [TS](#ts)         | Time Surface                  | 2016 | C++       | ✓   |      |
| [KNoise](#knoise) | Khodamoradi's Noise           | 2018 | C++       | ✓   |      |
| [EvFlow](#evflow) | Event Flow                    | 2019 | C++       | ✓   |      |
| [YNoise](#ynoise) | Yang's Noise                  | 2020 | C++       | ✓   |      |
| [EDnCNN](#edncnn) | Event Denoising CNN           | 2020 | C++       |     | ✓    |
| [DWF](#dwf)       | Double Window Filter          | 2021 | C++       | ✓   |      |
| [MLPF](#mlpf)     | Multilayer Perceptron Filter  | 2021 | C++       |     | ✓    |
| [EvZoom](#evzoom) | Event Zoom                    | 2021 | Python    |     | ✓    |
| [GEF](#gef)       | Guided Event Filter           | 2021 | Python    | ✓   |      |
| [RED](#red)       | Recursive Event Denoisor      | -    | C++       | ✓   |      |

### Running by single file

You can run `eval_denoisor.py` to test one of the above denoising algorithms:

```bash
python eval_denoisor.py                     \
--file './data/demo/samples/demo-01.aedat4' \
--denoisor 'ynoise'                         
```

+ `--file` / `-f`: path of sequence data.
+ `--denoisor`: select a denoising algorithm. You can revise denoisor's 
parameters in `./configs/denoisors.py`.

NOTE: Some algorithms need to install libtorch in advance and 
[compile with cuda](#cuda).

### Running by datasets

You can run `eval_benchmark.py` to test all sequences store in `./data` folder.

```bash
python eval_benchmark.py  \
--input_path './data'     \
--output_path './result'  \
--denoisor 'ynoise' --store_result --store_score
```

+ `--input_path` / `-i`: path of the datasets folder.
+ `--output_path` / `-o`: path of saving denoising results.
+ `--denoisor`: select a denoising algorithm. You can revise denoisor's 
parameters in `./configs/denoisors.py`.
+ `--store_result`: turn on denoising result storing.
+ `--store_score`: turn on mean ESR score calculation.

NOTE: The structure of the dataset folder must meet the [requirements](#datasets).

## Building your own denoising benchmark

### Datasets <span id="datasets"></span>

Download our **Event Noisy Dataset (END)**, including [D-END](https://drive.google.com/file/d/1ZatTSewmb-j6RsrJxMWEQIE3Sm1yraK-/view?usp=sharing) (Daytime part) and [N-END](https://drive.google.com/file/d/17ZDhuYdtHui9nqJAfiYYX27omPY7Rpl9/view?usp=sharing) (Night part), then unzip and paste them into `./data` folder:

```
./data/
├── D-END
│   ├── nd00
│   │   ├── Architecture-ND00-1.aedat4
│   │   ├── Architecture-ND00-2.aedat4
│   │   ├── Architecture-ND00-3.aedat4
│   │   ├── Bicycle-ND00-1.aedat4
│   │   ├── Bicycle-ND00-2.aedat4
│   │   ├── ...
│   ├── nd04
│   │   ├── Architecture-ND04-1.aedat4
│   │   ├── Architecture-ND04-2.aedat4
│   │   ├── ...
│   ├── ...
├── N-END
│   ├── nd00
│   │   ├── ...
│   ├── ...
├── ...
```

Also you can paste your customized datasets into `./data` folder (only supported 
aedat4 file now). They should be rearranged as the following structure: 

```
./data/
├── <Your Dataset Name>
│   ├── Subclass-1
│   │   ├── Sequences-1.*
│   │   ├── Sequences-2.*
│   │   ├── ...
│   ├── Subclass-2
│   │   ├── Sequences-1.*
│   │   ├── Sequences-2.*
│   │   ├── ...
│   ├── ...
├── ...
```

### Algorithms

We provide a general template to facilitate building your own denoising algorithm, see `./configs/denoisors.py`:

```python
class your_denoisor:
    def __init__(self, resolution, 
                 modified_params: Dict, 
                 default_params: Dict) -> None:
        # /*-----------------------------------*/
        #         initialize parameters
        # /*-----------------------------------*/

    def accept(self, events):
        # /*-----------------------------------*/
        #   receive noise sequence and process
        # /*-----------------------------------*/
    
    def generateEvents(self):
        # /*-----------------------------------*/
        #   perform denoising and return result
        # /*-----------------------------------*/
```



## Citations

### Projects using this repo

This repository is derived from **[E-MLB: Multilevel Benchmark for Event-Based Camera Denoising](https://ieeexplore.ieee.org/document/10078400)**.

```bibtex
@article{ding2023mlb,
  title     = {E-MLB: Multilevel Benchmark for Event-Based Camera Denoising},
  author    = {Ding, Saizhe and Chen, Jinze and Wang, Yang and Kang, Yu and Song, Weiguo and Cheng, Jie and Cao, Yang},
  journal   = {IEEE Transactions on Multimedia},
  year      = {2023},
  publisher = {IEEE}
}
```

### References

<span id="ts"></span>
**Time Surface** &nbsp; Hots: a hierarchy of event-based time-surfaces for pattern recognition

```bibtex
@article{lagorce2016hots,
  title     = {Hots: a hierarchy of event-based time-surfaces for pattern recognition},
  author    = {Lagorce, Xavier and Orchard, Garrick and Galluppi, Francesco and Shi, Bertram E and Benosman, Ryad B},
  journal   = {IEEE transactions on pattern analysis and machine intelligence},
  pages     = {1346--1359},
  year      = {2016},
  publisher = {IEEE}
}
```

<span id="knoise"></span>
**KNoise** &nbsp; O(N)-Space Spatiotemporal Filter for Reducing Noise in Neuromorphic Vision Sensors

```bibtex
@article{khodamoradi2018n,  
  title     = {O(N)-Space Spatiotemporal Filter for Reducing Noise in Neuromorphic Vision Sensors},
  author    = {Khodamoradi, Alireza and Kastner, Ryan},
  journal   = {IEEE Transactions on Emerging Topics in Computing},
  year      = {2018},
  publisher = {IEEE}
}
```

<span id="evflow"></span>
**EvFlow** &nbsp; EV-Gait: Event-based robust gait recognition using dynamic vision sensors

```bibtex
@inproceedings{wang2019ev,
  title     = {EV-Gait: Event-based robust gait recognition using dynamic vision sensors},
  author    = {Wang, Yanxiang and Du, Bowen and Shen, Yiran and Wu, Kai and Zhao, Guangrong and Sun, Jianguo and Wen, Hongkai},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {6358--6367},
  year      = {2019}
}
```

<span id="ynoise"></span>
**YNoise** &nbsp; Event density based denoising method for dynamic vision sensor

```bibtex
@article{feng2020event,
  title     = {Event density based denoising method for dynamic vision sensor},
  author    = {Feng, Yang and Lv, Hengyi and Liu, Hailong and Zhang, Yisa and Xiao, Yuyao and Han, Chengshan},
  journal   = {Applied Sciences},
  year      = {2020},
  publisher = {MDPI}
}
```

<span id="edncnn"></span>
**EDnCNN** &nbsp; Event probability mask (epm) and event denoising convolutional neural network (edncnn) for neuromorphic cameras

```bibtex
@inproceedings{baldwin2020event,  
  title     = {Event probability mask (epm) and event denoising convolutional neural network (edncnn) for neuromorphic cameras},
  author    = {Baldwin, R and Almatrafi, Mohammed and Asari, Vijayan and Hirakawa, Keigo},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {1701--1710},
  year      = {2020}
}
```

<span id="dwf"></span>
<span id="mlpf"></span>
**DWF & MLPF** &nbsp; Low Cost and Latency Event Camera Background Activity Denoising

```bibtex
@article{guo2022low,  
  title     = {Low Cost and Latency Event Camera Background Activity Denoising},
  author    = {Guo, Shasha and Delbruck, Tobi},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year      = {2022},
  publisher = {IEEE}
}
```

<span id="evzoom"></span>
**EvZoom** &nbsp; EventZoom: Learning to denoise and super resolve neuromorphic events

```bibtex
@inproceedings{duan2021eventzoom,
  title     = {EventZoom: Learning to denoise and super resolve neuromorphic events},
  author    = {Duan, Peiqi and Wang, Zihao W and Zhou, Xinyu and Ma, Yi and Shi, Boxin},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {12824--12833},
  year      = {2021}
}
```

<span id="gef"></span>
**GEF** &nbsp; Guided event filtering: Synergy between intensity images and neuromorphic events for high performance imaging

```bibtex
@article{duan2021guided,
  title     = {Guided event filtering: Synergy between intensity images and neuromorphic events for high performance imaging},
  author    = {Duan, Peiqi and Wang, Zihao W and Shi, Boxin and Cossairt, Oliver and Huang, Tiejun and Katsaggelos, Aggelos K},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year      = {2021},
  publisher = {IEEE}
}
```

## Acknowledgement
Special thanks to [Yang Wang](mailto:ywang120@ustc.edu.cn).
