import os
from builtins import print

import distance
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm


matrix = []
movies = []

movie_index_map = {}
index_movie_map = {}

bayes = GaussianNB()

removed_indexes_from_user29 = [140, 422, 1004, 1585, 1763, 1992, 2363, 3002, 4050, 4651]


def init():
    global matrix
    global movies

    movies = get_movies("../data/small_movies.npy")
    matrix = get_matrix("../data/small_matrix.npy")

    print('Matrix shape: ' + str(matrix.shape))

    # avg_rate_columns = []
    # for index in range(matrix.shape[1]):
    #     avg = average(matrix[:, index])
    #     avg_rate_columns.append([avg, index])
    #
    # arr = np.array(avg_rate_columns)
    #
    # sorted_avgs = arr[arr[:, 0].argsort()[::-1]][0:10]

    # for i in range(42, matrix.shape[0]):
    #     if os.path.isfile("../output/output_"+str(i)+".csv"):
    #         continue
    #   t = threading.Thread(target=check_movie_liking_for_user, args=(i, ))
    #   t.start()

    make_random_rates_zero()

    check_movie_liking_for_user_2_factors(29)  # most important function
    check_movie_liking_for_user_3_factors(29)  # most important function

    # up = user_preference(0, 2)
    # ia = item_acceptance(0, 2)
    # fi = friend_inference(0, 2)
    # output = round((up + ia + fi) / 3, 4)
    # print(up)
    # print(ia)
    # print(fi)
    #
    # print(output)


def make_random_rates_zero():
    global matrix
    rated_number = {}
    for i, row in enumerate(matrix):
        num = 0
        for index, rates in enumerate(row):
            if rates == 5:
                num = num + 1
        rated_number[i] = num

    print(rated_number)

    fives_indexes = []
    for index, rates in enumerate(matrix[29]):
        if rates == 5:
            fives_indexes.append(index)

    for ind in removed_indexes_from_user29:
        matrix[29][ind] = 0.0

    print(fives_indexes)
    print('Number of fives before: '+str(len(fives_indexes)))

    fives_indexes = []
    for index, rates in enumerate(matrix[29]):
        if rates == 5:
            fives_indexes.append(index)

    print('Number of fives after: ' + str(len(fives_indexes)))


def check_movie_liking_for_user_2_factors(user_row_id):
    global movies
    likings = []

    print('Running 2 factor code for user: ' + str(user_row_id))
    for movie_index in tqdm(range(movies.shape[0])):
        if matrix[29][movie_index] == 0:
            liking = check_movie_liking_2(user_row_id, int(movie_index)), \
                     int(movie_index), \
                     movies[int(movie_index)][1]

            likings.append(liking)

    likings = np.array(likings)
    likings = likings[likings[:, 0].argsort()[::-1]]

    with open('../output_2_factors/output_' + str(user_row_id) + '.csv', 'w') as the_file:
        the_file.write('movie_row_index,probability,movie_title' + '\n')
        for like in likings:
            # print('Probability: ' + str(like[0]) + ' - ' + str(like[2]))
            the_file.write(str(like[1]) + ',' + str(like[0]) + ',' + str(like[2]) + '\n')


def check_movie_liking_for_user_3_factors(user_row_id):
    global movies
    likings = []
    print('Running 3 factor code for user: ' + str(user_row_id))
    for movie_index in tqdm(range(movies.shape[0])):
        if matrix[29][movie_index] == 0:
            liking = check_movie_liking_3(user_row_id, int(movie_index)), \
                     int(movie_index), \
                     movies[int(movie_index)][1]

            likings.append(liking)

    likings = np.array(likings)
    likings = likings[likings[:, 0].argsort()[::-1]]

    with open('../output_3_factors/output_'+str(user_row_id)+'.csv', 'w') as the_file:
        the_file.write('movie_row_index,probability,movie_title' + '\n')
        for like in likings:
            # print('Probability: ' + str(like[0]) + ' - ' + str(like[2]))
            the_file.write(str(like[1]) + ',' + str(like[0]) + ',' + str(like[2]) + '\n')





def check_movie_liking_2(user_row_id, movie_column_id):
    up = user_preference(user_row_id, movie_column_id)
    ia = item_acceptance(user_row_id, movie_column_id)
    fi = 0  # friend_inference(user_row_id, movie_column_id)
    output = round((up + ia + fi) / 3, 4)
    # output = up * ia * fi
    # print('Liking of the user: '+str(user_row_id) + ' of the movie: '+str(movie[1]) + ' is: ' + str(output))

    return output


def check_movie_liking_3(user_row_id, movie_column_id):
    up = user_preference(user_row_id, movie_column_id)
    ia = item_acceptance(user_row_id, movie_column_id)
    fi = friend_inference(user_row_id, movie_column_id)
    output = round((up + ia + fi) / 3, 4)
    # output = up * ia * fi
    # print('Liking of the user: '+str(user_row_id) + ' of the movie: '+str(movie[1]) + ' is: ' + str(output))

    return output


def similar_movies(matrix_column):
    global movies

    similar = []  # [similarity, matrix_number, movie_id, movie_name]
    # temp_movies = np.delete(movies, matrix_column, axis=0)
    for i, id_movie_genres in enumerate(movies):
        if id_movie_genres[0] == index_movie_map[matrix_column]:
            active_column = id_movie_genres[2].split('|')
            current_column = movies[int(matrix_column), 2].split('|')
            similarity = 1 - distance.jaccard(active_column, current_column)
            # print(str(active_column) + ' - ' + str(current_column) + ' - ' + str(similarity))
            similar.append([similarity, i, id_movie_genres[0], id_movie_genres[1]])

    similar = np.array(similar)
    return similar[similar[:, 0].argsort()[::-1]]


def user_preference(matrix_row, matrix_column):  # first factor
    global matrix

    temp_matrix = np.copy(matrix)
    temp_matrix[:] *= 2
    # movie_rate = temp_matrix[matrix_row, matrix_column]

    temp_matrix = np.delete(temp_matrix, matrix_column, 1)

    # print('Movie rate: '+str(movie_rate))

    similar = np.array(similar_movies(matrix_column)[0:10])
    similar = similar[similar[:, 1].astype(int).argsort()[::-1]]

    return user_preference_for_similar_movies(temp_matrix, similar, matrix_row)


def user_preference_for_similar_movies(temp_matrix, similar, matrix_row):

    temp_matrix = np.delete(temp_matrix, similar[:, 1], 1)

    output = temp_matrix[matrix_row, :].astype(int)

    temp_matrix = np.delete(temp_matrix, matrix_row, 0)
    columns = []
    for index in range(temp_matrix.shape[1]):
        columns.append(temp_matrix[:, index].astype(int))

    bayes.fit(columns, output)

    temp_matrix = matrix[:, similar[:, 1].astype(int)]
    temp_matrix[:] *= 2
    temp_matrix = np.delete(temp_matrix, matrix_row, 0)
    columns = []
    for index in range(temp_matrix.shape[1]):
        columns.append(temp_matrix[:, index].astype(int))

    # output = output = temp_matrix[matrix_row, :]

    prediction = np.array([bayes.predict(columns)]).T
    ideal_movie = np.copy(prediction)
    ideal_movie[:] = 10
    # similar = np.append(similar, prediction, axis=1) # for displaying similarities

    output = f1_score(prediction, ideal_movie, average='micro')

    return output


def item_acceptance(matrix_row, matrix_column):  # second factor
    # obuci model na svima koji nisu uzeti u razmatranje
    # predict na ovima sto su uzeti
    global matrix
    temp_matrix = np.copy(matrix)
    temp_matrix[:] *= 2

    temp_matrix = np.delete(temp_matrix, matrix_row, 0)

    similar_users = []  # [similarity, matrix_number]
    for index, user in enumerate(temp_matrix):
        if index != matrix_row:
            similar_users.append([pearsonr(temp_matrix[index, :], temp_matrix[matrix_row, :])[0], index])

    similar_users = np.array(similar_users)

    similar_users = similar_users[similar_users[:, 0].argsort()[::-1]][
                    0:100]  # we take first 100 users as "friends" of active user

    return item_accaptance_for_similar_users(temp_matrix, similar_users, matrix_column)


def item_accaptance_for_similar_users(temp_matrix, similar_users, matrix_column):
    temp_matrix = np.delete(temp_matrix, similar_users[:, 1], 0)

    output = temp_matrix[:, matrix_column].astype(int)
    temp_matrix = np.delete(temp_matrix, matrix_column, 1)
    rows = []
    for index in range(temp_matrix.shape[0]):
        rows.append(temp_matrix[index, :].astype(int))

    bayes.fit(rows, output)

    temp_matrix = matrix[similar_users[:, 1].astype(int), :]
    temp_matrix[:] *= 2
    temp_matrix = np.delete(temp_matrix, matrix_column, 1)
    rows = []
    for index in range(temp_matrix.shape[0]):
        rows.append(temp_matrix[index, :].astype(int))

    # output = output = temp_matrix[matrix_row, :]

    prediction = np.array([bayes.predict(rows)]).T
    ideal_movie = np.copy(prediction)
    ideal_movie[:] = 10
    # similar = np.append(similar, prediction, axis=1) # for displaying similarities

    output = f1_score(prediction, ideal_movie, average='micro')
    return output


def friend_inference(matrix_row, matrix_column):  # third factor
    global matrix
    temp_matrix = np.copy(matrix)
    temp_matrix[:] *= 2

    similar_users = []  # [similarity, matrix_number]
    for index, user in enumerate(temp_matrix):
        if index != matrix_row:
            similar_users.append([pearsonr(temp_matrix[index, :], temp_matrix[matrix_row, :])[0], index])

    similar_users = np.array(similar_users)
    similar_users = similar_users[similar_users[:, 0].argsort()[::-1]][
                    0:10]  # we take first 10 users as "friends" of active user

    similar = np.array(similar_movies(matrix_column)[0:10])
    similar = similar[similar[:, 1].astype(int).argsort()[::-1]]

    likings = []
    for sm in similar_users:
        # up = user_preference(sm[1], matrix_column)
        # likings.append(user_preference(int(sm[1]), matrix_column))
        likings.append(user_preference_for_similar_movies(temp_matrix, similar, int(sm[1])))

    avg = np.mean(likings)

    return avg


def get_movies(movies_location):
    global movies

    if os.path.isfile(movies_location):
        movies = np.load(movies_location)

    for i, movie in enumerate(movies):
        movie_index_map[movie[0]] = i
        index_movie_map[i] = movie[0]

    print('Movies shape: ' + str(movies.shape))
    return movies


def get_matrix(matrix_location):
    global matrix
    if os.path.isfile(matrix_location):
        matrix = np.load(matrix_location)
    else:
        print('Matrix array does not exists!')
    return matrix

init()
