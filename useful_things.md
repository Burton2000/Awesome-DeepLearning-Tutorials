## Installing NCCL and caffe ##
- http://www.nvidia.com/object/caffe-installation.html

## Install nvidia toolkit on Linux (Ubuntu 16.04) (installs an old driver at the same time)
- Download the CUDA Toolkit 8.0 https://developer.nvidia.com/cuda-80-ga2-download-archive
- sudo shutdown -r 0
- press ctrl+alt+f4 to bring up terminal then login
- sudo service lightdm stop
- Follow all instructions found on this link: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions
- Add the following to your .bashrc:

`export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}$` 

`export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`

## Install latest nvidia drivers on Linux (Ubuntu 16.04)
- Download the latest driver from here http://www.nvidia.co.uk/Download/index.aspx?lang=uk selecting Linux 64 bit
- sudo shutdown -r 0
- press ctrl+alt+f4 to bring up terminal then login
- sudo service lightdm stop
- sudo sh NVIDIA-Linux-x86_64-384.81_linux.run
- accept and say yes to everything
- sudo shutdown -r 0

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
(This link for the Tensorflow package might change so you can find the latest version of it here: https://www.tensorflow.org/install/install_windows)

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
`conda install -c peterjc123 pytorch`
- Windows 7/8, Cuda 8 
`conda install -c peterjc123 pytorch_legacy=0.1.12`

ref: https://github.com/pytorch/pytorch/issues/494

## Install Anaconda on ubuntu ##
- `mkdir conda` 
- `cd conda` (create and go to a new directory for us to download anaconda install scripts to)
- `wget https://repo.continuum.io/archive/Anaconda3-5.0.0.1-Linux-x86_64.sh` (get latest link from https://www.anaconda.com/download/#linux)
- `bash Anaconda3-5.0.0.1-Linux-x86_64.sh`
- This will give an ouput where you need to press `ENTER` to continue and then keep pressing `ENTER` to read through the license.
- Agree to the license by typing `yes`
- Install location will be shown now, if you are happy with it press `ENTER` to continue or change to something else
- Install will take a bit of time but once its finished it will ask if you want to update your bashrc with the anaconda path, you should type `yes` so you can use the `conda` command.
- `source ~/.bashrc` to refresh your updataed bashrc
- Check its all installed by typing `conda list` this should show all the packages available to you

## Create Anaconda environments on ubuntu ##
- `conda create --name my_env python=3` you can leave out the pyton argument if you dont want to specify exactly what python version
- `source activate my_evn` to activate your environment, your command prompt prefix should change to include the name of your environment

## Install Tensorflow Ubuntu ##
https://www.tensorflow.org/install/install_linux#InstallingAnaconda

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

Then if you are using QT you can add these to your QT .pro file: 

- `- INCLUDEPATH += /usr/include/hdf5/serial`
- `- LIBS += -L/usr/lib/x86_64-linux-gnu/`

## g++ basic compiler options
http://web.engr.oregonstate.edu/~rubinma/Mines_274/Content/Slides/05_compilation.pdf

## Codec for video compression in virtualdub
- https://www.videohelp.com/software/ffdshow

## Python

Matplotlib remove whitespace in a figure (Domagoj):
- https://stackoverflow.com/questions/9295026/matplotlib-plots-removing-axis-legends-and-white-spaces

scipy.misc.imresize float/uint8 problem:
- https://github.com/scipy/scipy/issues/4458

## error: Unable to find vcvarsall.bat ##

Encounted this issue when trying to use Cython in Windows. This is caused by not have a c/c++ compiler installed. Quickest solution is to install this:

http://landinghub.visualstudio.com/visual-cpp-build-tools

You can also install visual studio 2015 and should fix the problem.

## Ubuntu batch rename files

- `for f in *.png; do mv "$f" "${f#image}"; done` (in this example rename all *.png removing 'image' from their name)

## Ubuntu find out disk space

- `du -h --max_length=1` will give size of all things in current folder
- `df -h ` will give overall free/used space on each disk

## Ubuntu mounting network drives issue

If you mount a network filesystem and the folders when mounted are empty or you get an error like: "wrong fs type, bad option, bad superblock" then you may just need to install cifs-utils. If mount.cifs is not in your /sbin then do 

`sudo apt-get install cifs-utils`

https://askubuntu.com/questions/525243/why-do-i-get-wrong-fs-type-bad-option-bad-superblock-error

## Tensorflow windows cupti dll missing for tracing
- When you use CUDA 8.0, the file cupti64_80.dll lies in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\extras\CUPTI\libx64. I just fixed the problem by copying the dll into C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin, and the file cupti.lib in the same location into C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64. And it works!

