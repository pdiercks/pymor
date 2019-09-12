#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""plot surface of

    s(x; µ) = 1 / sqrt(
        (x0 - µ0)**2 + (x1 - µ1)**2 + 0.1**2
    )
    for µ in P=range ** 2 over Ω =[0.1, 0.9]**2

Usage:
    plot_gaussian_surface.py [options] GRID NSAMPLES
    plot_gaussian_surface.py -h | --help

Arguments:
    GRID                Number of points to be used in x,y direction.
    NSAMPLES            Size of training set.

Options:
    --range=R           Range of parameter space given as string
                        containing a tuple of floats. [default: (-1.0, -0.01)]
    -h --help           Show this message.
"""

import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from docopt import docopt

# pymor
from pymor.parameters.base import ParameterType
from pymor.basic import *

def parse_arguments(args):
    args = docopt(__doc__, args)
    args['GRID'] = int(args['GRID'])
    args['NSAMPLES'] = int(args['NSAMPLES'])
    args['--range'] = eval(args['--range'])
    assert isinstance(args['--range'], tuple)
    return args

def main(args):
    args = parse_arguments(args)
    # create mesh
    x = np.linspace(0.1, 0.9, num=args['GRID'])
    y = x.copy()
    X, Y = np.meshgrid(x, y)

    # function s
    def s(x, y, mu):
        return 1.0 / np.sqrt(
            (x - mu["x"])**2 + (y - mu["y"])**2 + 0.1**2
        )

    # parametrization
    parameter_type = ParameterType({"x": (), "y": ()})
    parameter_space = CubicParameterSpace(
        parameter_type, ranges={"x": args['--range'],
                                "y": args['--range']})

    training_set = parameter_space.sample_randomly(args['NSAMPLES'])
    for (fid, mu) in enumerate(training_set):
        fig = plt.figure(fid)
        ax = fig.add_subplot(111, projection='3d')
        Z = s(X, Y, mu)
        ax.plot_surface(X, Y, Z, cmap="autumn_r")
        ax.contour(X, Y, Z, cmap="autumn_r", linestyles="-", offset=0)
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
