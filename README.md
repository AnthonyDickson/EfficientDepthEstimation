# Efficient Depth Estimation
This code repository contains the code used our publications "User-centred Depth Estimation Benchmarking for VR Content Creation from Single Images" [1] and "Benchmarking Monocular Depth Estimation Models for VR Content Creation from a User Perspective" [2].

The code is separated into three sections:
* Efficient Depth Estimation [ReSIDE/](./ReSIDE)
* Benchmarking Depth Estimation Models [Benchmark/](./Benchmark)
* Our User Study on Amazon Mechanical Turk [MTurk/](./MTurk)

## Getting Started

-   Download the trained models: [Depth estimation networks](https://drive.google.com/file/d/1QaUkdOiGpMuzMeWCGbey0sT0wXY0xtsj/view?usp=sharing) <br>
-   Download the data (only necessary for training): [NYU-v2 dataset](https://drive.google.com/file/d/1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw/view?usp=sharing) <br>
-   Create the conda environment:
    ```shell script
    conda env create -f environment.yml
    ```
-   Activate the conda environment:
    ```shell script
    conda activate ReSIDE
    ```

### Demo
```shell script
python -m demo
```
### Test
```shell script
python -m test
```
### Training
```shell script
python -m train
```

## Acknowledgements
This repo includes code from:
* [Revisiting Single Image Depth Estimation](https://github.com/JunjH/Revisiting_Single_Depth_Estimation.git)
* [Consistent Video Depth Estimation](https://github.com/facebookresearch/consistent_depth.git)

## Citing Our Work
If you use the code, please cite:
```
@inproceedings{dickson2021benchmarking,
  title={Benchmarking Monocular Depth Estimation Models for VR Content Creation from a User Perspective},
  author={Dickson, Anthony and Knott, Alistair and Zollmann, Stefanie},
  booktitle={2021 36th International Conference on Image and Vision Computing New Zealand (IVCNZ)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```

## References
1. Dickson, A., Knott, A., Zollmann, S., Lee, S.-h., Okabe, M., and Wuensche, B. (2021). User-centred Depth Estimation Benchmarking for VR Content Creation from Single Images. In Pacific Graphics Short Papers, Posters, and Work-in-Progress Papers. The Eurographics Association
2. Dickson, A., Knott, A., and Zollmann, S. (2021). Benchmarking Monocular Depth Estimation Models for VR Content Creation from a User Perspective. In 2021 36th International Conference on Image and Vision Computing New Zealand (IVCNZ), 1–6. IEEE