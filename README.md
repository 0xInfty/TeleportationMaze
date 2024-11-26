# Maze with Teleportation

## About

This is a repository created for an assignment from the COMPSCI5087 Artificial Intelligence M course from University of Glasgow.

The idea is to apply a discrete state-space search algorithm to solve a 2D maze with wormholes.

## Installation

1. Create and activate a Python 3.14 environment with Cython
```conda create -n AICoursework python=3.14 cython; conda activate AICoursework```
2. Install Mazelib
```pip install mazelib --use-pep517```
3. Install all other required packages
```pip install numpy networkx jupyter pandas matplotlib pillow Image ipythonblocks```
4. Download the AIMA toolbox from the course [Moodle](https://moodle.gla.ac.uk/mod/resource/view.php?id=4652387)
5. Clone this repository
6. Update the `AIMA_TOOLBOX_ROOT` variable inside the `plotter.py` module.
7. Good to go :D

## Authors and acknowledgment

This repository was created and updated exclusively by Valeria Rocio Pais Malcalza, student at University of Glasgow.

My most sincere thanks to the lecturers for the base activities and code provided, used as the basis for the development of this code. 

Credits to the AIMA toolbox for the ready-to-go search tools too, since they are the fundamental bricks upon both the lecturers' and this repository's code were developed.