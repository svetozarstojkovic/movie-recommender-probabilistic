import numpy as np
import os.path
import csv

import tqdm as tqdm
from scipy import spatial, math


users = []
rating = []
movies = []
ratingMap = {}
userIndexMap = {}
indexUserMap = {}
movieIndexMap = {}
indexMovieMap = {}
movieLink = {}

matrix = np.zeros((1, 1))


def init_data():
    # fill_movies()
    # fill_users()

    set_matrix()

    rate_movies()


def fill_movies():
    global movies

    if os.path.isfile("../data/movies.npy"):
        movies = np.load("../data/movies.npy")

    for i, movie in enumerate(movies):
        movieIndexMap[movie[0]] = i
        indexMovieMap[i] = movie[0]

    print('Movies shape: ' + str(movies.shape))


def fill_users():
    global rating
    global users

    if os.path.isfile("../data/users.npy"):
        users = np.load("../data/users.npy")
        rating = np.load("../data/rating.npy")
        k = 0
        temp = []
        for rate in rating:
            if rate[0] not in temp and rate[0] != '':
                userIndexMap[rate[0]] = k
                indexUserMap[k] = rate[0]
                temp.append(rate[0])
                k = k + 1

    print('Users shape:' + str(users.shape))
    print('Ratings shape:' + str(rating.shape))


def rate_movies():
    for rt in tqdm.tqdm(rating):
        rate_movie(rt[0], rt[1], rt[2])


def rate_movie(user_id, movie_id, rate):
    matrix[userIndexMap[user_id], movieIndexMap[movie_id]] = rate
    np.save("../data/matrix", matrix)


def set_matrix():
    global matrix
    matrix = np.zeros((len(users), len(movies)))
    if os.path.isfile("../data/matrix.npy"):
        matrix = np.load("../data/matrix.npy")
    else:
        np.save("../data/matrix", matrix)
    print('Matrix shape: '+str(matrix.shape))


def loginUser(username, password):
    for user in users:
        if np.logical_and(user.username == username, user.password == password):
            return user
    return None


def get_user(username):
    for user in users:
        if user.username == username:
            return user

    return None


def get_movies_for_username(username):
    index = userIndexMap.get(username)
    return matrix[index, :]


def get_movie_name_rating(username):
    movieMap = {}
    ratings = get_movies_for_username(username)
    for id, rate in enumerate(ratings):
        mn = indexMovieMap.get(id).decode("utf8")
        movieMap[mn] = rate

    movieMap = sorted(movieMap.items(), key=np.operator.itemgetter(1), reverse=True)
    return movieMap


def movies_i_watched(username):
    movieMap = {}
    ratings = get_movies_for_username(username)
    for id, rate in enumerate(ratings):
        if rate != 0:
            mn = indexMovieMap.get(id)
            movieMap[mn] = rate

    movieMap = sorted(movieMap.items(), key=np.operator.itemgetter(1), reverse=True)
    return movieMap


def get_movies():
    return indexMovieMap


def get_movies_for_similar(myUsername, similar_users):
    topMovies = []
    retTopMovies = []
    myMovies = get_movies_for_username(myUsername)
    newUser = True
    for username, value in similar_users:
        if np.logical_and(abs(value) >= 0.0001, not np.isnan(value)):
            newUser = False
            break

    for username, value in similar_users:
        if np.logical_and(abs(value) >= 0.0001, not np.isnan(value)):
            hisMovies = get_movies_for_username(username)
            hisMovies = np.array(hisMovies, dtype=np.int)
            for myIndex, tempValue in enumerate(hisMovies):  # his movies are set to 0 if i watched them
                if np.logical_and(myMovies[myIndex] != 0, hisMovies[myIndex] != 0):
                    hisMovies[myIndex] = 0
            for num, hisRate in enumerate(hisMovies):
                if hisRate != 0:
                    if indexMovieMap.get(num) not in topMovies:
                        topMovies.append(indexMovieMap.get(num))
                        retTopMovies.append([indexMovieMap.get(num), float("{0:.3f}".format(value))])
                    elif indexMovieMap.get(num) in topMovies:
                        index = topMovies.index(indexMovieMap.get(num))
                        retMovieValue = retTopMovies[index]
                        if value > retMovieValue[1]:
                            retTopMovies.pop(index)
                            retTopMovies.insert(index, [indexMovieMap.get(num), float("{0:.3f}".format(value))])

    print(len(retTopMovies))
    return retTopMovies


def jaccard_similarity(username):
    print("Username: " + str(username))
    global matrix
    similar_users = {}
    my_movies = get_movies_for_username(username)
    my_movies = np.array(my_movies)
    my_movies_zero = np.where(my_movies != 0)[0]
    print(matrix)
    for idx, row in enumerate(matrix):
        print(str(idx) + ' row: ' + str(row))
        union = 0
        intersection = 0
        # for i, value in enumerate(row):
        #     if np.logical_and(value != 0, my_movies[i] != 0):
        #         union = union+1
        #         intersection = intersection+1
        #     elif np.logical_and(value != 0, my_movies[i] == 0):
        #         union = union+1
        #     elif np.logical_and(value == 0, my_movies[i] != 0):
        #         union = union+1

        rowArray = np.array(row)
        row_non_zero = np.where(rowArray != 0)[0]
        print("Row: " + str(row_non_zero))
        print("My zero:" + str(my_movies_zero))
        intersection = len(np.intersect1d(row_non_zero, my_movies_zero))
        union =  len(np.unique(np.concatenate((row_non_zero, my_movies_zero), axis=0)))

        if union != 0:
            similarity = float(intersection)/union
            similar_users[indexUserMap.get(idx)] = similarity
        else:
            similar_users[indexUserMap.get(idx)] = 0.0


    similar_users.pop(username)

    similar_users = sorted(similar_users.items(), key=np.operator.itemgetter(1), reverse=True)
    print('SimilarUsers: ' + str(similar_users))

    movieList = get_movies_for_similar(username, similar_users)
    # print "JaccardMovieList:" + str(movieList)
    return movieList


def cosine_similarity(username):
    print("Username: " + str(username))
    global matrix
    similar_users = {}
    print(matrix.shape)
    my_movies = get_movies_for_username(username)
    for idx, row in enumerate(matrix):
        # print str(idx) + ' row: ' + str(row)
        similarity = 1 - spatial.distance.cosine(my_movies, row)
        similar_users[indexUserMap.get(idx)] = similarity

    similar_users.pop(username)

    similar_users = sorted(similar_users.items(), key=np.operator.itemgetter(1), reverse=True)
    print('SimilarUsers: ' + str(similar_users))

    movieList = get_movies_for_similar(username, similar_users)
    # print 'CosineMovieList: '+ str(movieList)
    return movieList

def centered_cosine_similarity(username):
    print("Username: " + str(username))
    global matrix
    similar_users = {}
    print(matrix.shape)
    my_movies = center_row(get_movies_for_username(username))
    print(my_movies)
    for idx, row in enumerate(matrix):
        # print str(idx) + ' row: ' + str(row)

        rowArray = center_row(row)

        similarity = 1 - spatial.distance.cosine(my_movies, rowArray)
        similar_users[indexUserMap.get(idx)] = similarity

    similar_users.pop(username)

    similar_users = sorted(similar_users.items(), key=np.operator.itemgetter(1), reverse=True)
    print('SimilarUsers: ' + str(similar_users))

    movieList = get_movies_for_similar(username, similar_users)
    # print 'CosineMovieList: '+ str(movieList)
    return movieList

def center_row(row):
    rowArray = np.array(row)
    row_mean = rowArray[rowArray.nonzero()].mean()
    for rowId, val in enumerate(rowArray):
        if val != 0:
            rowArray[rowId] = val - row_mean
    return rowArray

def pearson_similarity(username):
    my_movies = get_movies_for_username(username)
    similar_users = {}
    for idx, row in enumerate(matrix):
        similarity = pearson_def(my_movies, row)
        similar_users[indexUserMap.get(idx)] = similarity

    similar_users.pop(username)

    similar_users = sorted(similar_users.items(), key=np.operator.itemgetter(1), reverse=True)
    print('SimilarUsers: ' + str(similar_users))

    movieList = get_movies_for_similar(username, similar_users)
    print('PearsonMovieList: ' + str(movieList))
    return movieList

def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)


def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff
    # print math.sqrt(xdiff2 * ydiff2)
    if math.sqrt(xdiff2 * ydiff2) != 0:
        return diffprod / math.sqrt(xdiff2 * ydiff2)
    else:
        return 0


init_data()