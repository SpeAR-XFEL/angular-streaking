# Angular streaking simulation framework

Prerequisites: Git, Python 3.6+, Numpy, Scipy, Matplotlib, pip.

First, clone the repository (go to the directory you want the project to be in):

```
git clone git@github.com:larsfu/angular_streaking.git
```

Then use 
```
python -m pip install -e angular_streaking
```
to make the streaking package available to your Python installation.
The GUI script is available through
```
python angular_streaking/simulations/gui.py
```
and allows playing around with all the parameters. To start understanding the code, a good starting point
is `examples/simple.py`, where a very basic simulation using the library is implemented.
