## Video Representation Learning by Dense Predictive Coding 

This repository contains the implementation of Dense Predictive Coding (DPC). 

Links: [[Arxiv](https://arxiv.org/abs/1909.04656)] [[Video](https://youtu.be/43KIHUvHjB0)] [[Project page](http://www.robots.ox.ac.uk/~vgg/research/DPC/)]

![arch](asset/arch.png)

### Installation

The implementation should work with python >= 3.6, pytorch >= 0.4, torchvision >= 0.2.2. 

The repo also requires cv2 (`conda install -c menpo opencv`), tensorboardX >= 1.7 (`pip install tensorboardX`), joblib, tqdm, ipdb.

### Prepare data

Follow the instructions [here](process_data/).

### Self-supervised training (DPC)

Change directory `cd DPC/dpc/`

* example: train DPC-RNN using 2 GPUs, with 3D-ResNet18 backbone, on Kinetics400 dataset with 128x128 resolution, for 300 epochs
  ```
  python main.py --gpu 0,1 --net resnet18 --dataset k400 --batch_size 128 --img_dim 128 --epochs 300
  ```

* example: train DPC-RNN using 4 GPUs, with 3D-ResNet34 backbone, on Kinetics400 dataset with 224x224 resolution, for 150 epochs
  ```
  python main.py --gpu 0,1,2,3 --net resnet34 --dataset k400 --batch_size 44 --img_dim 224 --epochs 150
  ```

### Evaluation: supervised action classification

Change directory `cd DPC/eval/`

* example: finetune pretrained DPC weights (replace `{model.pth.tar}` with pretrained DPC model)
  ```
  python test.py --gpu 0,1 --net resnet18 --dataset ucf101 --batch_size 128 --img_dim 128 --pretrain {model.pth.tar} --train_what ft --epochs 300
  ```

* example (continued): test the finetuned model (replace `{finetune_model.pth.tar}` with finetuned classifier model)
  ```
  python test.py --gpu 0,1 --net resnet18 --dataset ucf101 --batch_size 128 --img_dim 128 --test {finetune_model.pth.tar}
  ```

### DPC-pretrained weights

It took us **more than 1 week** to train the 3D-ResNet18 DPC model on Kinetics-400 with 128x128 resolution, and it tooks about **6 weeks** to train the 3D-ResNet34 DPC model on Kinetics-400 with 224x224 resolution (with 4 Nvidia P40 GPUs). 

Download link: [3D-ResNet18-Kinetics400-128x128](https://drive.google.com/file/d/1jbMg2EAX8armIQA6_0YwfATh_h7rQz4u/view?usp=sharing), [3D-ResNet34-Kinetics400-224x224](https://drive.google.com/file/d/1d2XhuUwGTgEBg2cKkQbfJG8omHaSlELZ/view?usp=sharing)

* example: finetune `3D-ResNet34-Kinetics400-224x224`
  ```
  python test.py --gpu 0,1 --net resnet34 --dataset ucf101 --batch_size 44 --img_dim 224 --pretrain {model.pth.tar} --train_what ft --epochs 300
  ```

### Citation

If you find the repo useful for your research, please consider citing our paper: 
```
@article{han2019dpc,
  title={Video Representation Learning by Dense Predictive Coding},
  author={Han, Tengda and Xie, Weidi and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1909.04656},
  year={2019}
}
```
For any questions, welcome to create an issue or contact Tengda Han ([htd@robots.ox.ac.uk](mailto:htd@robots.ox.ac.uk)).



