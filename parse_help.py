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
        for i in range(len(args.dimensions) / 3):
            start = i * 3
            end = start + 3
            dimensions.append(args.dimensions[start:end])

    return tuple(dimensions)