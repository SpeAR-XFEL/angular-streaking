# Angular streaking simulation framework

Prerequisites: Git, Python 3.6+, Numpy, Scipy, Matplotlib, pip.

To start developing, first clone the repository (go to the directory you want the project to be in first):

```
git clone git@github.com:larsfu/angular_streaking.git
```

Then use 
```
python -m pip install -e angular_streaking
```
to make the streaking package available to your python installation.
A good starting point is to now start the GUI script

```
python angular_streaking/simulations/gui.py
```
and play around a little. To start developing, a good starting point
may be `examples/simple.py`, where a very basic simulation using the 
library is implemented.