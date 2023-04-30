# E-MLB: Multilevel Benchmark for Event-Based Camera Denoising

The simple benchmark for event-based denoising. Any questions please contact me with [KugaMaxx@outlook.com](mailto:KugaMaxx@outlook.com).

<span id="animation"></span>
![animation](https://raw.githubusercontent.com/KugaMaxx/cuke-emlb/main/assets/images/animation.gif "animation")



## Installation

Make sure that your device meets:
  + Ubuntu with [Cmake](https://cmake.org/) ≥ 3.5.1 and Python ≥ 3.8.
  + [DV](https://inivation.gitlab.io/dv/dv-docs/docs/getting-started/) and [LibTorch](https://pytorch.org/) are optimal.


### Dependencies

Install dependencies, including [Pybind11](https://pybind11.readthedocs.io/en/stable/) (for python), [Boost](https://www.boost.org/) and [OpenBlas](https://www.openblas.net/) (for optimized performance):
```bash
sudo apt-get install python3-dev python3-pybind11
sudo apt-get install libboost-dev
sudo apt-get install libopenblas-dev
```


### Build from source

Create a new virtual environment (recommended):
```bash
conda create -n emlb python=3.10
conda activate emlb
```

First, clone this repo and build by `setup.sh`:
```bash
sh setup.sh
```


### Running a demo

Run `eval_demo.py` to test:
```bash
python eval_demo.py
```

Then you will receive the visualization results as shown in [gif](#animation).



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

You can run `eval_denoisor.py` to test one of the above denoising algorithms:

```bash
python eval_denoisor.py --denoisor knoise
```

+ `--input_path` / `-i`: path of the datasets folder.
+ `--output_path` / `-o`: path of denoising results. If not set, it will not be saved to disk.
+ `--denoisor` / `-d`: select a denoising algorithm which can be found in `./configs/denoisors.py`. Some algorithms need to be compiled after installing LibTorch, please refer to [CUDA compile](#cuda) for details.
+ `--excl_hotpixel`: decide whether to remove hot pixels in advance.


### DV support

If you want to test above modules in DV software, please install the following additional package and then recompile `setup.sh`:

```bash
sudo apt-get install dv-runtime-dev
```

You can add the path where this project located in and you will find the list of available modules. More details please refer to the [DV's official guide](https://inivation.gitlab.io/dv/dv-docs/docs/first-module/). It should be noted that we have not solved the problem of CNN method running in DV.


### CUDA compile <span id="cuda"></span>

When installed LibTorch, you can compile `setup.sh` with `<path/to/libtorch>` as follows:

```bash
sh setup.py -l <path/to/libtorch>/share/cmake/Torch/
```

Download [pretrained models](https://drive.google.com/drive/folders/1BytQnsNRlv1rJyMotElIqklOz1oCt2Vd?usp=sharing) and paste them into `./modules/_net/` folder.



## Building your own denoising benchmark

### Datasets

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

Also you can paste your customized datasets into `/data` folder(supported file types can be checked at [here](https://github.com/KugaMaxx/taro-dvstoolkit)). They should be rearranged as the following structure: 

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
class your_denoisor(BaseDenoisor):
    def __init__(self, model=<your_denoisor>, **params=<paramters>):
        super().__init__()
        self.model = model
        self.params = params

    def run(self, data: evtool.dtype):
        # /*-----------------------------------*/
        #   input denoising implementation here
        # /*-----------------------------------*/
        return data
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
