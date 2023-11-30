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
This tests the pretrained ResNet50 model from [5].
### Training
```shell script
python -m train
```
By default, this trains the model with pretrained ResNet50 encoder with the hu2018 decoder for 20 epochs.

Note that this code uses Weights & Biases (wandb) to cache model weights and example outputs.
If you get an error about credentials and want to run the code without syncing to wandb, set the following environment variable before running the script again: `WANDB_MODE=dryrun`.
Run outputs can be found under the folder [wanbd/](./wandb) using a name that starts with either `run` or `dryrun`, followed by timestamp and a random ID.

## Pretrained Models
| Encoder             | Decoder           | DEL1  | REL   | Size (MB) | URL                                                                                             |
|---------------------|-------------------|-------|-------|-----------|-------------------------------------------------------------------------------------------------|
| EfficientNet-B0 [3] | Hu et al. [5]     | 0.816 | 0.140 | 20.6      | https://github.com/AnthonyDickson/EfficientDepthEstimation/releases/download/v1.0.0/ENB0-HU.pth |
| EfficientNet-B4 [3] | Hu et al. [5]     | 0.840 | 0.128 | 75.5      | https://github.com/AnthonyDickson/EfficientDepthEstimation/releases/download/v1.0.0/ENB4-HU.pth |
| ResNet50        [4] | Hu et al. [5]     | 0.843 | 0.125 | 258       | https://github.com/AnthonyDickson/EfficientDepthEstimation/releases/download/v1.0.0/RN50-HU.pth |
| EfficientNet-B0 [3] | Ranftl et al. [6] | 0.807 | 0.144 | 14.9      | https://github.com/AnthonyDickson/EfficientDepthEstimation/releases/download/v1.0.0/ENB0-LR.pth |
| EfficientNet-B4 [3] | Ranftl et al. [6] | 0.835 | 0.130 | 66.1      | https://github.com/AnthonyDickson/EfficientDepthEstimation/releases/download/v1.0.0/ENB4-LR.pth |
| ResNet50        [4] | Ranftl et al. [6] | 0.849 | 0.124 | 156       | https://github.com/AnthonyDickson/EfficientDepthEstimation/releases/download/v1.0.0/RN50-LR.pth |

Place the pretrained models in a folder called `checkpoints` in the project root directory.

## Acknowledgements
This repo includes code from:
* [Revisiting Single Image Depth Estimation](https://github.com/JunjH/Revisiting_Single_Depth_Estimation.git) [5]
* [MiDaS](https://github.com/isl-org/MiDaS/releases/tag/v1) [6]
* [Consistent Video Depth Estimation](https://github.com/facebookresearch/consistent_depth.git) [7]

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
3. Tan, M., & Le, Q. (2019, May). Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning (pp. 6105-6114). PMLR.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
5. Hu, J., Ozay, M., Zhang, Y., & Okatani, T. (2019, January). Revisiting single image depth estimation: Toward higher resolution maps with accurate object boundaries. In 2019 IEEE winter conference on applications of computer vision (WACV) (pp. 1043-1051). IEEE.
6. Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2020). Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, 44(3), 1623-1637.
7. Luo, X., Huang, J. B., Szeliski, R., Matzen, K., & Kopf, J. (2020). Consistent video depth estimation. ACM Transactions on Graphics (ToG), 39(4), 71-1.