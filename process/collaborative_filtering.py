import csv
import os

import numpy as np
from builtins import print
from scipy.stats import pearsonr

matrix = []
movies = []

movie_index_map = {}
index_movie_map = {}


def init():
    global matrix
    global movies

    movies = get_movies("../data/small_movies.npy")
    matrix = get_matrix("../data/small_matrix.npy")

    print('Movies shape: ' + str(movies.shape))
    print('Matrix shape: ' + str(matrix.shape))

    get_collaborative_probability_for_user(29)
    # for i in range(60, 62):
    #     remove_rated(i)


def remove_rated(user_row):
    with open('../output/output_'+str(user_row)+'.csv', 'r') as f:
        rec_movies = list(csv.reader(f))

    user_movies = matrix[user_row]
    watched = []
    for i, movie in enumerate(rec_movies):
        if i == 0:
            continue
        if user_movies[int(movie[0])] != 0:
            watched.append(i)

    watched = sorted(watched, reverse=True)

    for index in watched:
        for i, movie in enumerate(rec_movies):
            if i == 0:
                continue
            if int(movie[0]) == index:
                del rec_movies[index]

    with open('../output_without_watched/output_' + str(user_row) + '.csv', 'w') as the_file:
        for movie in rec_movies:
            the_file.write(','.join(movie) + '\n')

    print('Done for user: ' +str(user_row))


def get_collaborative_probability_for_user(matrix_row):
    global matrix
    global movies

    similar_users = get_similar_users(matrix_row)

    temp_matrix = np.copy(matrix)

    complete_movies_list = []

    indexes = []

    for i, user in enumerate(similar_users):
        print(i)
        row = temp_matrix[int(user[1])]
        max_movies_indexes = np.argsort(row)[::-1][:1000]
        for index in max_movies_indexes:
            if index not in indexes:
                if row[index] > 0:
                    complete_movies_list.append([user, index, row[index], round(float(user[0]) * row[index], 4)])
                    indexes.append(index)

    complete_movies_list = np.array(complete_movies_list)
    complete_movies_list = complete_movies_list[complete_movies_list[:, 3].argsort()[::-1]]

    with open('../output_collaborative/output_' + str(matrix_row) + '.csv', 'w') as the_file:
        the_file.write('movie_index,user_index,probability,movie_rate,factor,movie_title' + '\n')
        for probability_index_rate in complete_movies_list:
            the_file.write(str(probability_index_rate[1]) + ',' +
                           str(int(probability_index_rate[0][1])) + ',' +
                           str(round(probability_index_rate[0][0], 4)) + ',' +
                           str(probability_index_rate[2]) + ',' +
                           str(probability_index_rate[3]) + ',' +
                           str(movies[probability_index_rate[1]][1]) + '\n')


def get_similar_users(matrix_row):
    similar_users = []  # [similarity, matrix_number]
    temp_matrix = np.copy(matrix)
    for index, user in enumerate(temp_matrix):
        if index != matrix_row:
            similar_users.append([pearsonr(temp_matrix[index, :], temp_matrix[matrix_row, :])[0], int(index)])

    similar_users = np.array(similar_users)

    similar_users = similar_users[similar_users[:, 0].argsort()[::-1]]  # [:100]

    return similar_users


def get_movies(movies_location):
    global movies

    if os.path.isfile(movies_location):
        movies = np.load(movies_location)

    for i, movie in enumerate(movies):
        movie_index_map[movie[0]] = i
        index_movie_map[i] = movie[0]


    return movies


def get_matrix(matrix_location):
    global matrix
    if os.path.isfile(matrix_location):
        matrix = np.load(matrix_location)
    else:
        print('Matrix array does not exists!')
    return matrix


init()
