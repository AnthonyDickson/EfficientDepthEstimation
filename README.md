# Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps with Accurate Object Boundaries
Junjie Hu, Mete Ozay, Yan Zhang, Takayuki Okatani [https://arxiv.org/abs/1803.08673](https://arxiv.org/abs/1803.08673)

## Results
![](https://github.com/junjH/Revisiting_Single_Depth_Estimation/raw/master/examples/example.png)
![](https://github.com/junjH/Revisiting_Single_Depth_Estimation/raw/master/examples/results.png)

## Running

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

## Citation
If you use the code or the pre-processed data, please cite:
```
@inproceedings{Hu2018RevisitingSI,
  title={Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps With Accurate Object Boundaries},
  author={Junjie Hu and Mete Ozay and Yan Zhang and Takayuki Okatani},
  booktitle={IEEE Winter Conf. on Applications of Computer Vision (WACV)},
  year={2019}
}
```