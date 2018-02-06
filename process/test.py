import csv

from builtins import print

import numpy as np

removed_indexes_from_user29 = [140, 422, 1004, 1585, 1763, 1992, 2363, 3002, 4050, 4651]


def init():
    print('Collaborative evaluation:\t\t\t\t' + str(get_median_for_user29('../output_collaborative/output_29.csv')))
    print('Simplified probabilistic evaluation:\t' + str(get_median_for_user29('../output_2_factors/output_29.csv')))
    print('Full probabilistic evaluation:\t\t\t' + str(get_median_for_user29('../output_3_factors/output_29.csv')))


def get_median_for_user29(location):
    with open(location, 'r') as f:
        lines = list(csv.reader(f))

    found_indexes = []

    for index in removed_indexes_from_user29:
        for i, line in enumerate(lines):
            if i == 0:
                continue
            if int(line[0]) == index:
                found_indexes.append(i)
    print('\nValues: '+str(sorted(found_indexes)))
    return np.median(found_indexes)


init()
