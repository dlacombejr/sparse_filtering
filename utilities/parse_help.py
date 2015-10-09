# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 23:40:41 2015

@author: dan
"""


def parse_dims(args):

    dimensions = []

    if args.convolution == 'n':
        for i in range(len(args.dimensions) / 2):
            start = i * 2
            end = start + 2
            dimensions.append(args.dimensions[start:end])

    if args.convolution == 'y':
        for i in range(len(args.dimensions) / 4):
            start = i * 4
            end = start + 4
            dimensions.append(args.dimensions[start:end])

    return tuple(dimensions)


def parse_iter(args):

    iterations = []

    for i in range(len(args.iterations)):
        iterations.append(args.iterations[i])

    return iterations


def parse_models(args):

    models = []

    for i in range(len(args.model)):
        models.append(args.model[i])

    return models