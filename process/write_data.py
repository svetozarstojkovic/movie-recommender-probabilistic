import csv
import os

import numpy as np
from builtins import print
from tqdm import tqdm

movies = []
rating_map = {}
user_index_map = {}
index_user_map = {}

movie_index_map = {}
index_movie_map = {}

matrix = []


def init():
    fill_small_movies()
    fill_small_users()


def fill_movies():
    global movies
    movies = []

    with open('../dataset/movie.csv', 'r') as f:
        reader = csv.reader(f)
        movies = np.array(list(reader))
        movies = np.delete(movies, 0, 0)

    for i, movie in enumerate(movies):
        movie_index_map[movie[0]] = i
        index_movie_map[i] = movie[0]

    np.save("../data/movies", movies)


def fill_users():
    rating = []
    users = []

    with open('../data/rating.csv', 'r') as f:
        reader = csv.reader(f)
        print('Reading rating file')
        rating = list(reader)
        rating = np.delete(rating, 0, 0)
        print('Done reading rating file')

    current_user = None
    for rate in tqdm(rating):
        if rate[0] not in rating_map:
            rating_map[rate[0]] = []
            rating_map[rate[0]].append(rate[1]+'_'+rate[2])
            if current_user is None:
                current_user = rate[0]
            elif current_user != rate[0]:
                current_movies = np.zeros(len(movies))
                for rated_movie in rating_map[current_user]:
                    current_movies[movie_index_map[rated_movie.split('_')[0]]] = rated_movie.split('_')[1]
                matrix.append(current_movies)
                current_user = rate[0]
        else:
            rating_map[rate[0]].append(rate[1]+'_'+rate[2])

        # if rate[0] not in users and rate[0] != '':
        #     user_index_map[rate[0]] = k
        #     index_user_map[k] = rate[0]
        #     users.append(rate[0].strip())
        #     k = k + 1

    users = np.array(users)

    np.save("../data/matrix", matrix)
    # np.save("../data/rating", rating)
    # np.save("../data/users", users)


def fill_small_movies():
    global movies
    movies = []

    with open('../dataset/small/movies.csv', 'r') as f:
        reader = csv.reader(f)
        movies = np.array(list(reader))
        movies = np.delete(movies, 0, 0)

    for i, movie in enumerate(movies):
        movie_index_map[movie[0]] = i
        index_movie_map[i] = movie[0]

    np.save("../data/small_movies", movies)


def fill_small_users():
    rating = []
    users = []

    with open('../data/small_rating.csv', 'r') as f:
        reader = csv.reader(f)
        print('Reading rating file')
        rating = list(reader)
        rating = np.delete(rating, 0, 0)
        print('Done reading rating file')

    current_user = None
    for rate in tqdm(rating):
        if rate[0] not in rating_map:
            rating_map[rate[0]] = []
            rating_map[rate[0]].append(rate[1]+'_'+rate[2])
            if current_user is None:
                current_user = rate[0]
            elif current_user != rate[0]:
                current_movies = np.zeros(len(movies))
                for rated_movie in rating_map[current_user]:
                    current_movies[movie_index_map[rated_movie.split('_')[0]]] = rated_movie.split('_')[1]
                matrix.append(current_movies)
                current_user = rate[0]
        else:
            rating_map[rate[0]].append(rate[1]+'_'+rate[2])

        # if rate[0] not in users and rate[0] != '':
        #     user_index_map[rate[0]] = k
        #     index_user_map[k] = rate[0]
        #     users.append(rate[0].strip())
        #     k = k + 1

    # users = np.array(users)

    np.save("../data/small_matrix", matrix)
    # np.save("../data/rating", rating)
    # np.save("../data/users", users)


init()
