# Angular streaking simulation framework

Prerequisites: Git, [Poetry](https://python-poetry.org/docs/)

First, clone the repository (go to the directory you want the project to be in):

```
git clone git@github.com:larsfu/angular_streaking.git
```

Optionally, set poetry to create the Python virtual environment inside the project folder (this allows IDEs like VS Code to find it):
```
poetry config virtualenvs.in-project true
```
Install all dependencies and create the virtual environment using `poetry install`, then either use `poetry shell` to run the simulations or select the environment inside your editor/IDE.

The GUI script is available in `simulations/gui.py` and allows playing around with all the parameters. To start understanding the code, a good starting point
is `examples/simple.py`, where a rudimentary simulation is implemented.
