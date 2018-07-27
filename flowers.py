#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import math
import random
from collections import namedtuple

Measurement = namedtuple(
    'Measurement', (
        'color',
        'length',
        'width',
    )
)

MEASUREMENTS = [
    Measurement('r', 3.0, 1.5),
    Measurement('b', 2.0, 1.0),
    Measurement('r', 4.0, 1.5),
    Measurement('b', 3.0, 1.0),
    Measurement('r', 3.5, 0.5),
    Measurement('b', 2.0, 0.5),
    Measurement('r', 5.5, 1.0),
    Measurement('b', 1.0, 1.0),
]

TestData = namedtuple(
    'TestData', (
        'length',
        'width',
        'expected_color',
    )
)

TEST_DATA = [
    TestData(4.5, 1.0, 'r'),
    TestData(3.7, 0.8, 'r'),
    TestData(4.5, 1.5, 'r'),
    TestData(1.8, 0.7, 'b'),
    TestData(2.5, 0.5, 'b'),
]


def main():
    # random.seed(12345)
    (w1, w2, b) = train()
    print('TEST DATA')
    for test_data in TEST_DATA:
        estimated_color = estimate_color(test_data.length, test_data.width,
                                         w1, w2, b)
        print(
            'length = {}, width = {} => estimated color = {} ({:.2f}) <=> '
            'expected color = {}'.format(
                test_data.length, test_data.width,
                to_color(estimated_color), estimated_color,
                test_data.expected_color))


def train():
    print('TRAINING DATA')

    best_w1 = w1 = random.uniform(-10, 10)
    best_w2 = w2 = random.uniform(-10, 10)
    best_b = b = random.uniform(-10, 10)

    min_sum_error_squared = 9999999999.9
    for i in xrange(10000):
        w1 = random.uniform(-10, 10)
        w2 = random.uniform(-10, 10)
        b = random.uniform(-10, 10)
        sum_error_squared = 0.0
        for measurement in MEASUREMENTS:
            estimated_color = estimate_color(
                measurement.length, measurement.width, w1, w2, b)
            error_squared = (
                estimated_color - color_to_number(measurement.color)) ** 2
            sum_error_squared += error_squared
            # print(
            #     'length = {}, width = {} => estimated color = {} ({:.2f}) <=> '
            #     'actual color = {} [err = {}]'.format(
            #         measurement.length, measurement.width,
            #         to_color(estimated_color), estimated_color,
            #         measurement.color, error_squared))
        if sum_error_squared < min_sum_error_squared:
            min_sum_error_squared = sum_error_squared
            best_w1 = w1
            best_w2 = w2
            best_b = b
            print(
                'Found better parameters: w1 = {}, w2 = {}, b = {} [{}]'
                .format(w1, w2, b, min_sum_error_squared))
    return (best_w1, best_w2, best_b)


def estimate_color(length, width, w1, w2, b):
    return sigmoid(w1 * length + w2 * width + b)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def to_color(x):
    return 'b' if x <= 0.5 else 'r'


def color_to_number(color):
    return 0 if color == 'b' else 1


if __name__ == "__main__":
    main()
