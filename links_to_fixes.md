## Installing NCCL and caffe ##
- http://www.nvidia.com/object/caffe-installation.html

## Anaconda environments and running spyder ## 
- https://conda.io/docs/using/envs.html#change-environments-activate-deactivate
- http://stackoverflow.com/questions/30170468/how-to-run-spyder-in-virtual-environment

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

## Ubuntu batch rename files

- `for f in *.png; do mv "$f" "${f#image}"; done` 
