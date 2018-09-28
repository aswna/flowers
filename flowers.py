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
    test(w1, w2, b)


def train():
    learning_rate = 0.02
    print('TRAINING DATA')
    w1 = random.uniform(-10, 10)
    w2 = random.uniform(-10, 10)
    b = random.uniform(-10, 10)
    for i in xrange(10000):
        sum_error_squared = 0
        for measurement in MEASUREMENTS:
            z = w1 * measurement.length + w2 * measurement.width + b
            # use z in predict_color?
            predicted_color = predict_color(
                measurement.length, measurement.width, w1, w2, b)
            target_color = color_to_number(measurement.color)

            error_squared = (predicted_color - target_color) ** 2
            sum_error_squared += error_squared

            derror_squared_dpredicted_color = 2 * (predicted_color -
                                                   target_color)

            dpredicted_color_dz = sigmoid(z) * (1 - sigmoid(z))

            dz_dw1 = measurement.length
            dz_dw2 = measurement.width
            dz_db = 1

            derror_squared_dw1 = (
                derror_squared_dpredicted_color * dpredicted_color_dz * dz_dw1)
            derror_squared_dw2 = (
                derror_squared_dpredicted_color * dpredicted_color_dz * dz_dw2)
            derror_squared_db = (
                derror_squared_dpredicted_color * dpredicted_color_dz * dz_db)

            w1 -= learning_rate * derror_squared_dw1
            w2 -= learning_rate * derror_squared_dw2
            b -= learning_rate * derror_squared_db

        if i % 1000 == 0:
            print('sum_error_squared = {}'.format(sum_error_squared))
    return (w1, w2, b)


def test(w1, w2, b):
    print('TEST DATA')
    for test_data in TEST_DATA:
        predicted_color = predict_color(test_data.length, test_data.width,
                                        w1, w2, b)
        print(
            'length = {}, width = {} => predicted color = {} ({:.2f}) <=> '
            'expected color = {}'.format(
                test_data.length, test_data.width,
                to_color(predicted_color), predicted_color,
                test_data.expected_color))


def predict_color(length, width, w1, w2, b):
    return sigmoid(w1 * length + w2 * width + b)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def slope_of_sigmoid_numerically(x):
    step_size = 0.01
    return (sigmoid(x + step_size) - sigmoid(x)) / step_size


def to_color(x):
    return 'b' if x <= 0.5 else 'r'


def color_to_number(color):
    return 0 if color == 'b' else 1


if __name__ == "__main__":
    main()
