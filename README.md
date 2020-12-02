# PRoGAN: Pose Randomization for Weakly Paired Image Style Translation

This is the official repository for PRoGAN. The currently released code is for reviewers of RA-L to reproduce the experimental results on *AEROGROUND DATASET* and *WEAKPAIRED DATASET*.

## PS: We apologize to the reviewers that we accidentally uploaded a wrong source code file in the supplementary folder along with the manuscript, and therefore we decided to open-source this project ahead of time so that the reviewer is able to repreduce the result!!!

## Dependencies

There are a few dependencies required to run the code.  They are listed below:

### System Environments:

`Python 3.7`

`CUDA 10.1`

`CuDnn`

### Pip dependencies:

`torch>=1.6.0`

`torchvision>=0.7.0`

`dominate>=2.4.0`

`visdom>=0.1.8.8`

`Kornia`

`TensorboardX`

`opencv-python`

`matplotlib`

`torchviz`



You can install these dependencies by changing your current directory to the PRoGAN directory and running:

``` shell
pip install -r requirements.txt  
```

Or manually install the dependencies, and Conda or Virtualenv is recommended to use.

#### Using Pre-trained model

Download the model here:


https://v2.fangcloud.com/share/09582731e63b1008d819c199eb

## Getting Started

### Datasets

If you want to train the Aeroground dataset or Carla dataset: 

- Aeroground: https://github.com/ZJU-Robotics-Lab/OpenDataSet

- WeakPaired: https://github.com/ZJU-Robotics-Lab/OpenDataSet



### Prepare your own dataset:

``` shell
mkdir datasets
cd datasets
# The dataset structure is shown as following:
├──Your dataset
  ├── trainA (realA images)
      ├── Image0001.jpg 
      └── ...
  ├── trainB (realB images)
      ├── Image0001.jpg
      └── ...
  ├── testA (testing realA images)
      ├── Image3000.jpg
      └── ... 
  ├── testb (testing realB images)
      ├── Image3000.jpg
      └── ... 
```

## Example Usage

### Training

If you want to train the PRoGAN network, then run:

``` shell
# on aeroground
python main.py --phase train --aeroground
# on weakPaired
python main.py --phase train --carla
# on other dataset
python main.py --phase train --name training_name --batch_size 1  --netG resnet_9blocks --load_size 256 --dataset your_dataset --input_nc your_image_channel --output_nc your_image_channel 
```

#### Training visualize

To visualize the training process, you can run:

``` shell
# use visdom
python -m visdom.server # http://localhost:8097
# use tensorboard
python tensorboard --logdir checkpoints/log/your_training_name/
```

#### Training options

By default, this will train the network on the `stereo_to_aerial ` dataset with the batch size of 1, learning rate of 0.00015 and run on GPU 0. There are several settings you can change by adding arguments below:

| Arguments           | What it will trigger                            | Default              |
| ------------------- | ----------------------------------------------- | -------------------- |
| --gpu_ids           | The ids of gpu to use                           | 0                    |
| --checkpoints_dir   | The place to save models                        | './checkpoints/'     |
| --input_nc          | The channel of input image                      | 3                    |
| --output_nc         | The channel of output image                     | 3                    |
| --batch_size        | The batch size of input                         | 1                    |
| --load_size         | The size of input image for network (128 / 256) | 128                  |
| --continue_train    | Continue to train                               |                      |
| --epoch             | The start epoch for continuing  to train        | 'latest'             |
| --phase             | Choose to train or validate (train / val)       | 'train'              |
| --lr                | The learning rate for training                  | 0.00015              |
| --train_writer_path | Where to write the Log of training              | './checkpoints/log/' |
| --val_writer_path   | Where to save the images of validating          | './outputs/'         |
| --aeroground        | Use dataset: AeroGround                         |                      |
| --carla             | Use dataset: WeakPaired                         |                      |

### Validating

To validate on the dataset, you could run:

``` shell
# on aeroground
python main.py --phase val --aeroground
# on your dataset
python main.py --phase val --name training_name --batch_size 1  --netG resnet_9blocks --load_size 256 --dataset your_dataset --input_nc your_image_channel --output_nc your_image_channel --epoch your_test_epoch
```

Then your test images will be output to `'./outputs/'` folder as default.

