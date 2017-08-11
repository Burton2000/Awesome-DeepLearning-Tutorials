## Installing NCCL and caffe ##
- http://www.nvidia.com/object/caffe-installation.html

## Adjust fan speed Nvidia gpu on Linux ##
In a terminal enter:

`nvidia-xconfig --enable-all-gpus`

`nvidia-xconfig --cool-bits=4`

Then restart your computer and nvidia x settings should have a gpu fan speed option.


## Caffe calculation of pool/conv spatial sizes ##
In caffe pooling and conv. operations are treated differently for output spatial size. 
- For pooing output size is ` top_size = ceil((bottom_size + 2*pad - kernel_size) / stride) + 1`
- For conv. output size is ` top_size = floor((bottom_size + 2*pad - kernel_size) / stride) + 1`
- https://github.com/BVLC/caffe/issues/1318

## Anaconda environments and running spyder ## 
- https://conda.io/docs/using/envs.html#change-environments-activate-deactivate
- http://stackoverflow.com/questions/30170468/how-to-run-spyder-in-virtual-environment

## Install tensorflow on Windows ##
To run Tensorflow on Windows you need python 3.5.2, the best way to install Tensorflow is to create an anaconda environment running 3.5.2. To create this environment and install Tensorflow in it do the following:

- Install Anaconda
- Open an Anaconda prompt and run the following
- `conda create --name nameofenv python=3.5.2 anaconda` (this creates a python 3.5.2 env. called nameofenv <- you can change this name to something else)
- Open an naconda prompt for your newly created environment (will be in the start menu) and run the following
- `pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.2.1-cp35-cp35m-win_amd64.whl ` 
- This link for the Tensorflow package might change so you can find the latest version of it here: https://www.tensorflow.org/install/install_windows

To use this created environment with Tensorflow within Pycharm we must change the Pycharm interpreter settings. This is done as follows:

- Open your Pycharm project 
- File -> Settings
- Click on Project Interpreter under Project:
- Click the Settings wheel then 'Add local'
- Find the python.exe for your created environment (this locaton can be found by opening an Anaconda prompt of your environment where it will tell you where to look)
- Select the python.exe of your created environment 
- Done!

## Install pytorch on Windows ##
You need an anaconda env of pytohn 3.6.1 and you will need cuda 8 installed.
- `conda create --name nameofenv python=3.6.1 anaconda` (this creates a python 3.6.1 env. called nameofenv <- you can change this name to something else)
- Open an anaconda prompt of your new environment and enter the following to installl pytorch depending on your os
- Windows 10, Cuda 8
`conda install -c peterjc123 pytorch=0.1.12`
- Windows 7/8, Cuda 8 
`conda install -c peterjc123 pytorch_legacy=0.1.12`

ref: https://github.com/pytorch/pytorch/issues/494

## Change home folder and remove token access in Jupyter Notebook ##
`jupyter notebook --ip=* --generate-config` 

This will generate a config file, edit this config file:

set:


- c.NotebookApp.notebook_dir = ''

to change the folder notebook opens with and set:

- c.NotebookApp.token = ''

to disable token requirement

these were lines 195 and 243 for me!

## Install hdf5 on ubuntu 16.04 ##
First:
- `sudo apt-get install libhdf5-*`  (installs libhdf5 libs)
- `sudo apt-get install hdf5-*` (install all other hdf5 stuff)

Then add these to your QT .pro file: 

- `- INCLUDEPATH += /usr/include/hdf5/serial`
- `- LIBS += -L/usr/lib/x86_64-linux-gnu/`


## Python

Matplotlib remove whitespace in a figure (Domagoj):
- https://stackoverflow.com/questions/9295026/matplotlib-plots-removing-axis-legends-and-white-spaces

## error: Unable to find vcvarsall.bat ##

Encounted this issue when trying to use Cython in Windows. This is caused by not have a c/c++ compiler installed. Quickest solution is to install this:

http://landinghub.visualstudio.com/visual-cpp-build-tools

You can also install visual studio 2015 and should fix the problem.

## Ubuntu batch rename files

- `for f in *.png; do mv "$f" "${f#image}"; done` (in this example rename all *.png removing 'image' from their name)

## Tensorflow windows cupti dll missing for tracing
- When you use CUDA 8.0, the file cupti64_80.dll lies in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\extras\CUPTI\libx64. I just fixed the problem by copying the dll into C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin, and the file cupti.lib in the same location into C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64. And it works!
