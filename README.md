# Efficient Depth Estimation
This code repository contains the code used our publications "User-centred Depth Estimation Benchmarking for VR Content Creation from Single Images" [1] and "Benchmarking Monocular Depth Estimation Models for VR Content Creation from a User Perspective" [2].

The code is separated into three sections:
* Efficient Depth Estimation [ReSIDE/](./ReSIDE)
* Benchmarking Depth Estimation Models [Benchmark/](./Benchmark)
* Our User Study on Amazon Mechanical Turk [MTurk/](./MTurk)

## Installation

-   Download the trained models: [Depth estimation networks](https://drive.google.com/file/d/1QaUkdOiGpMuzMeWCGbey0sT0wXY0xtsj/view?usp=sharing)
-   Download the data (only necessary for training): [NYU-v2 dataset](https://drive.google.com/file/d/1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw/view?usp=sharing)

### Local Installation
-   Create the conda environment:
    ```shell script
    conda env create -f environment.yml
    ```
-   Activate the conda environment:
    ```shell script
    conda activate ReSIDE
    ```
    
### Docker
You can use the Docker image `anthondickson/ede` or build the Docker image with the [Docker](./Docker) file.
Note that the Docker image uses PyTorch 1.7.1 instead of 1.3.1 which was used in the experiments in [1, 2].

## Usage Examples
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
By default, this trains the model with pretrained ResNet50 encoder with the hu2018 decoder for 20 epochs.

Note that this code uses Weights & Biases (wandb) to cache model weights and example outputs.
Run outputs can be found under the folder [wanbd/](./wandb) using a name that starts with either `run` or `dryrun`, followed by timestamp and a random ID.

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