#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
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
    print('TRAINING DATA')
    for measurement in MEASUREMENTS:
        estimated_color = estimate_color(measurement.length, measurement.width)
        print(
            'length = {}, width = {} => estimated color = {} ({:.2f}) <=> '
            'actual color = {}'.format(
                measurement.length, measurement.width,
                to_color(estimated_color), estimated_color,
                measurement.color))
    print('TEST DATA')
    for test_data in TEST_DATA:
        estimated_color = estimate_color(test_data.length, test_data.width)
        print(
            'length = {}, width = {} => estimated color = {} ({:.2f}) <=> '
            'expected color = {}'.format(
                test_data.length, test_data.width,
                to_color(estimated_color), estimated_color,
                test_data.expected_color))


def estimate_color(length, width):
    w1 = 0.9
    w2 = 0.5
    b = -3.2
    return sigmoid(w1 * length + w2 * width + b)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def to_color(x):
    return 'b' if x <= 0.5 else 'r'


if __name__ == "__main__":
    main()
