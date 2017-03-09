from pandas import Series
import math
import numpy as np
import pandas as pd

# the list of users
# each user's movie's rate need to be predicate
class TestUserMap:

    def __init__(self):
        self.map = {}

    def get_map(self):
        return self.map

    def put_user(self, user_id, test_user):
        self.map[user_id] = test_user

    def get_user(self, user_id):
        return self.map.get(user_id)


class TestUser:

    def __init__(self, user_id):
        self.user_id = user_id
        # the list of rated movie
        self.list_of_rated_movie = []
        # the list of rate of rated movie
        self.list_of_rate_of_rated_movie = []
        # the list of unrated movie
        self.list_of_unrated_movie = []

    def get_user_id(self):
        return self.user_id

    def get_list_of_rated_movie(self):
        return self.list_of_rated_movie

    def get_list_of_rate_of_rated_movie(self):
        return self.list_of_rate_of_rated_movie

    def get_list_of_unrated_movie(self):
        return self.list_of_unrated_movie

def build_test_user_map(test_file):
    '''
    build the test user list from test.txt
    :param test_file: test.txt
    :return: python dictionary
    '''
    df = pd.read_table(test_file, sep="\s+", header=None)

    rows = df.shape[0]

    # create a map to store the user id with TestUser object
    user_map = TestUserMap().get_map()

    for i in range(rows):
        user_id = df[0][i]

        if user_id not in user_map:
            user = TestUser(user_id)
            user_map[user_id] = user

        user = user_map[user_id]
        rate = df[2][i]
        if rate > 0:
            user.get_list_of_rated_movie().append(df[1][i])
            user.get_list_of_rate_of_rated_movie().append(df[2][i])
        else:
            user.get_list_of_unrated_movie().append(df[1][i])

    return user_map

def build_train_matrix(train_file, train_matrix):
    '''
    load the training data
    :param train_file: txt file
    :param matrix: python 2-d array
    :return: void
    '''
    file = open(train_file, "r")
    lines_of_file = file.read().strip().split("\n")

    for i in range(len(lines_of_file)):
        line = lines_of_file[i]
        train_matrix[i] = [int(val) for val in line.split()]

# num_of_users = 200
# num_of_movies = 1000
# train_matrix = [[0] * num_of_movies] * num_of_users
# build_train_matrix("train.txt", train_matrix)
# print(train_matrix[7 - 1][219 - 1] == 1)
# print(train_matrix[92 - 1][219 - 1] == 4)

def update_data_iuf(train_matrix, test_map):
    '''
    calculate the train matrix with iuf
    :param train_matrix: python 2-d array
    :param test_map: python dictionary
    :return: void
    '''
    # record the iuf of each movie
    iuf_map = {}

    # train matrix with iuf
    num_row = len(train_matrix)
    num_col = len(train_matrix[0])

    for c_idx in range(num_col):
        iuf = 0.0
        movie_id = c_idx + 1
        total_num_of_users = num_row

        # the list of user's rate which rate the given movie
        lst_rate = []

        for r_idx in range(num_row):
            user_id = r_idx + 1

            rate = train_matrix[user_id - 1][movie_id - 1]
            if rate > 0: lst_rate.append(rate)

        if len(lst_rate) != 0:
            iuf = math.log10(total_num_of_users / len(lst_rate))
        else:
            iuf = 1.0

        for r_idx in range(num_row):
            train_matrix[r_idx][c_idx] = iuf * train_matrix[r_idx][c_idx]

        iuf_map[movie_id] = iuf

    # test map with iuf
    for user_id, user in test_map.items():

        list_rated_movie = user.get_list_of_rated_movie()
        list_rate = user.get_list_of_rate_of_rated_movie()

        for i in range(len(list_rated_movie)):
            movie_id = list_rated_movie[i]
            movie_rate = list_rate[i]
            # update movie rate with iuf
            list_rate[i] = movie_rate * iuf_map[movie_id]

# num_of_users = 200
# num_of_movies = 1000
# train_matrix = [[0] * num_of_movies] * num_of_users
# build_train_matrix("train.txt", train_matrix)
# test_map = build_test_user_map("test5.txt")
# movie_id = 237
# user_id = 201
#
# print(test_map[user_id].get_list_of_rated_movie())
# print(test_map[user_id].get_list_of_rate_of_rated_movie())
#
# update_data_iuf(train_matrix, test_map)
#
# print(test_map[user_id].get_list_of_rated_movie())
# print(test_map[user_id].get_list_of_rate_of_rated_movie())

def avg_movie_rate_of_test_user(user_id, test_map):
    '''
    calculate the average rate of given test user
    :param user_id: int
    :param test_map: python dictionary
    :return: int
    '''
    user = test_map[user_id]
    list_of_rate_of_rated_movie = user.get_list_of_rate_of_rated_movie()

    avg_rate = 0.0
    if len(list_of_rate_of_rated_movie) != 0:
        avg_rate = sum(list_of_rate_of_rated_movie) / len(list_of_rate_of_rated_movie)

    return avg_rate

def avg_movie_rate_of_train_users(train_matrix):
    '''
    calculate the mean of each train user in the train data
    :param train_matrix: python 2-d array
    :return: python dictionary, K: train user id, V: mean of given train user
    '''

    map_mean_train_users = {}
    for index, row in enumerate(train_matrix):
        mean_rate = 0.0

        user_id = index + 1
        non_zero_list = [rate for rate in row if rate > 0]

        if len(non_zero_list) > 0:
            mean_rate = sum(non_zero_list) / len(non_zero_list)

        map_mean_train_users[user_id] = mean_rate

    return map_mean_train_users

# num_of_users = 200
# num_of_movies = 1000
# train_matrix = [[0] * num_of_movies] * num_of_users
# build_train_matrix("train.txt", train_matrix)
#
# non_zero_user = [rate for rate in train_matrix[2 - 1] if rate > 0]
# mean_user1 = sum(non_zero_user) / len(non_zero_user)
# map_mean = avg_movie_rate_of_train_users(train_matrix)
# print(map_mean[2] == mean_user1)

# def cal_inverse_user_frequency(movie_id, total_number_of_users, train_matrix):
#     '''
#     calculate the inverse user frequency for given user
#     :param movie_id: int
#     :return: float
#     '''
#     iuf = 0.0
#     count = 0
#
#     for index in range(total_number_of_users):
#         user_id = index + 1
#         if train_matrix[user_id - 1][movie_id - 1] != 0:
#             count += 1
#
#     if count != 0:
#         iuf = math.log10(total_number_of_users / count)
#     else:
#         iuf = 1
#
#     return iuf

# Cosine Similarity
def find_similar_neighbor_cosine(user_id, train_matrix, test_map):
    '''
    find the top k neighbors with cosine similarity
    :param user_id: int
    :param train_matrix: python 2-d array
    :param test_map: python dictionary
    :return: list_of_tuple(user_id, cosine_similarity)
    '''

    test_user = test_map[user_id]
    list_of_unrated_movie = test_user.get_list_of_unrated_movie()
    list_of_rated_movie = test_user.get_list_of_rated_movie()
    list_of_rate_of_rated_movie = test_user.get_list_of_rate_of_rated_movie()

    list_of_neighbor = []

    # find the neighbor
    # go through the 200 users
    for row in range(len(train_matrix)):
        train_user_id = row + 1

        # skip the target user itself
        if train_user_id == user_id: continue

        common_movie = 0

        numerator = 0.0
        denominator = 0.0
        cosine_similarity = 0.0

        sqr_sum_of_test_rate = 0.0
        sqr_sum_of_train_rate = 0.0

        for i in range(len(list_of_rated_movie)):
            movie_id = list_of_rated_movie[i]
            test_movie_rate = list_of_rate_of_rated_movie[i]
            train_movie_rate = train_matrix[train_user_id - 1][movie_id - 1]

            # movie rate with zero means the user doesn't rate the movie
            # we should not consider it as part of the calculation of cosine similarity
            if train_movie_rate != 0:
                common_movie += 1

                numerator += test_movie_rate * train_movie_rate
                sqr_sum_of_test_rate += math.pow(test_movie_rate, 2)
                sqr_sum_of_train_rate += math.pow(train_movie_rate, 2)

        denominator = math.sqrt(sqr_sum_of_test_rate) * math.sqrt(sqr_sum_of_train_rate)

        # the common movie between test data and train data should be larger than 1
        if common_movie > 1 and denominator != 0.0:
            cosine_similarity = numerator / denominator

            list_of_neighbor.append((train_user_id, cosine_similarity))

    list_of_neighbor.sort(key=lambda tup : tup[1], reverse=True)

    return list_of_neighbor

# num_of_users = 200
# num_of_movies = 1000
# train_matrix = [[0] * num_of_movies] * num_of_users
# build_train_matrix("train.txt", train_matrix)
# test_map = build_test_user_map("test5.txt")
# list_of_neighbors = find_similar_neighbor_cosine(201, train_matrix, test_map)
# print(list_of_neighbors)

def predict_rating_with_cosine_similarity(user_id, movie_id, num_of_neighbor, train_matrix, test_map, list_of_neighbors):
    '''
    predict the user's rating on the given movie based on cosine similarity
    :param user_id: int
    :param movie_id: int
    :param num_of_neighbor: int
    :param train_matrix: python 2-d array
    :param test_map: python dictionary
    :return: int
    '''
    # average rate of user in the test data
    avg_movie_rate_in_test = avg_movie_rate_of_test_user(user_id, test_map)

    numerator = 0.0
    denominator = 0.0

    counter = 0
    for i in range(len(list_of_neighbors)):
        # top k neighbors
        if counter > num_of_neighbor: break

        neighbor_id = list_of_neighbors[i][0]
        neighbor_similarity = list_of_neighbors[i][1]
        neighbor_movie_rate = train_matrix[neighbor_id - 1][movie_id - 1]

        if neighbor_movie_rate > 0:
            counter += 1

            # # iuf
            # iuf = cal_inverse_user_frequency(movie_id, len(train_matrix), train_matrix)
            # neighbor_movie_rate *= iuf

            # case amplification
            # p = 2.5
            # neighbor_similarity *= math.pow(math.fabs(neighbor_similarity), (p - 1))

            numerator += neighbor_similarity * neighbor_movie_rate
            denominator += neighbor_similarity

    if denominator != 0.0:
        result = numerator / denominator
    else:
        result = avg_movie_rate_in_test

    result = int(round(result))

    return result

# num_of_users = 200
# num_of_movies = 1000
# train_matrix = [[0] * num_of_movies] * num_of_users
# build_train_matrix("train.txt", train_matrix)
# test_map = build_test_user_map("test5.txt")
# build_train_matrix("train.txt", train_matrix)
# result = predict_rating_with_cosine_similarity(201, 1, 100, train_matrix, test_map)
# print(result == 4)
# result = predict_rating_with_cosine_similarity(300, 996, 100, train_matrix, test_map)
# print(result == 2)

# Pearson Correlation
def cal_pearson_correlation(test_users, test_mean_rate, train_users, train_mean_rate):
    '''
    calculate the pearson correlation
    :param test_users: list of tuple(movie id, movie rate) from unrated movie in test data
    :param test_mean_rate: mean of rate in the test user
    :param train_users: python list
    :param train_mean_rate: mean of rate in the train user
    :return: float
    '''
    # there are 3 cases:
    # case 1: result > 0
    #           return result
    # case 2.1: result == 0, because of no common component
    #           return 0
    # case 2.2: result == 0, because of in one or two of vector, each component == mean of vector
    #           return the mean of vector

    numerator = 0.0
    denominator = 0.0

    # filter the common components as vector
    vector_test_rates = []
    vector_train_rates = []

    for test_movie_id, test_movie_rate in test_users:
        train_movie_rate = train_users[test_movie_id - 1]
        if train_movie_rate > 0 and test_movie_rate > 0:
            vector_test_rates.append(test_movie_rate)
            vector_train_rates.append(train_movie_rate)

    #   no common component
    if len(vector_train_rates) == 0 or len(vector_test_rates) == 0:
        return 0.0

    # mean_test_rates = sum(vector_test_rates) / len(vector_test_rates)
    # mean_train_rates = sum(vector_train_rates) / len(vector_train_rates)

    adj_vector_test_users = [movie_rate - test_mean_rate for movie_rate in vector_test_rates]
    adj_vector_train_users = [movie_rate - train_mean_rate for movie_rate in vector_train_rates]

    s_adj_vector_test_users = Series(adj_vector_test_users)
    s_adj_vector_train_users = Series(adj_vector_train_users)

    numerator = Series.dot(s_adj_vector_test_users, s_adj_vector_train_users)
    denominator = math.sqrt(Series.dot(s_adj_vector_test_users, s_adj_vector_test_users)) * \
        math.sqrt(Series.dot(s_adj_vector_train_users, s_adj_vector_train_users))

    # each component of one or both vectors is the same
    if denominator == 0.0:
        return 0.0

    return numerator / denominator

# num_of_users = 200
# num_of_movies = 1000
# train_matrix = [[0] * num_of_movies] * num_of_users
# build_train_matrix("train.txt", train_matrix)
# test_map = build_test_user_map("test5.txt")
# build_train_matrix("train.txt", train_matrix)
#
# test_users = [(42, 4), (85, 2), (194, 5), (208, 5), (585, 1)]
# train_users = train_matrix[0]
#
# res = cal_pearson_correlation(test_users, train_users)
# print(res)

def find_similar_neighbor_pearson(user_id, train_matrix, test_map, train_mean_rate_map):
    '''
    calculate all neighbors's pearson correlation for given test user
    :param user_id: int
    :param train_df: python 2-d array
    :param test_map: python dictionary
    :param train_mean_rate_map: python dictionary, the mean of each train user
    :return: a list of tuple(user id, similarity)
    '''
    list_of_neighbors = []

    # average rate of given test user
    avg_movie_rate_in_test = avg_movie_rate_of_test_user(user_id, test_map)

    user = test_map[user_id]
    list_of_rated_movie = user.get_list_of_rated_movie()
    list_of_rate_of_rated_movie = user.get_list_of_rate_of_rated_movie()
    list_of_unrated_movie = user.get_list_of_unrated_movie()

    # zipped_list_of_rated_movie_with_rate = list(zip(list_of_rated_movie, list_of_rate_of_rated_movie))
    zipped_list_of_rated_movie_with_rate = []
    for i in range(len(list_of_rated_movie)):
        zipped_list_of_rated_movie_with_rate.append((list_of_rated_movie[i], list_of_rate_of_rated_movie[i]))

    # find the neighbor
    # go through the 200 users
    for index, row in enumerate(train_matrix):
        train_user_id = index + 1

        # skip the target user
        if train_user_id == user_id: continue

        # average rate of given train user
        avg_movie_rate_in_train = train_mean_rate_map[train_user_id]

        pearson_correlation = cal_pearson_correlation(zipped_list_of_rated_movie_with_rate, avg_movie_rate_in_test, row, avg_movie_rate_in_train)

        # correct the pearson correlation
        # range is [-1, 1]
        if pearson_correlation > 1.0:
            pearson_correlation = 1.0
        if pearson_correlation < -1.0:
            pearson_correlation = -1.0

        if pearson_correlation != 0.0:
            list_of_neighbors.append((train_user_id, pearson_correlation))

    # list_of_neighbors.sort(key=lambda tup: tup[1], reverse=True)

    return list_of_neighbors

# num_of_users = 200
# num_of_movies = 1000
# train_matrix = [[0] * num_of_movies] * num_of_users
# build_train_matrix("train.txt", train_matrix)
# test_map = build_test_user_map("test5.txt")
# build_train_matrix("train.txt", train_matrix)
# train_mean_rate_map = avg_movie_rate_of_train_users(train_matrix)
#
# test_user_id = 201
# list_of_neighbors = find_similar_neighbor_pearson(test_user_id,train_matrix,test_map,train_mean_rate_map)
# print(list_of_neighbors)
# print(len(list_of_neighbors))

def predict_rating_with_pearson_correlation(user_id, movie_id, train_matrix, test_map, train_mean_rate_map, list_of_neighbors):
    '''
    predict the rate on given user's movie
    :param user_id: int, test user id
    :param movie_id: int, test user's unrated movie id
    :param train_matrix: python 2-d array
    :param test_map: python dictionary
    :param list_of_neighbors: python list, list of tuple(train user id, pearson correlation)
    :return: int
    '''

    result = 0.0
    numerator = 0.0
    denominator = 0.0

    # list_of_neighbors = find_similar_neighbor_pearson(user_id, train_matrix, test_map, train_mean_rate_map)

    # the mean of given test user's movie rates
    test_mean_rate = avg_movie_rate_of_test_user(user_id, test_map)

    for neighbor in list_of_neighbors:
        train_user_id = neighbor[0]
        pearson_correlation = neighbor[1]

        # the mean of given train user's movie rates
        train_mean_rate = train_mean_rate_map[train_user_id]

        train_user_rate = train_matrix[train_user_id - 1][movie_id - 1]
        if train_user_rate > 0:

            # case amplification
            # p = 2.5
            # pearson_correlation *= math.pow(math.fabs(pearson_correlation), (p - 1))

            numerator += pearson_correlation * (train_user_rate - train_mean_rate)
            denominator += math.fabs(pearson_correlation)

    if denominator != 0.0:
        result = test_mean_rate + numerator / denominator
    else:
        result = test_mean_rate

    result = int(round(result))

    if result > 5:
        result = 5
    if result < 1:
        result = 1

    return result

# num_of_users = 200
# num_of_movies = 1000
# train_matrix = [[0] * num_of_movies] * num_of_users
# build_train_matrix("train.txt", train_matrix)
# test_map = build_test_user_map("test5.txt")
# build_train_matrix("train.txt", train_matrix)
#
# user_id = 251
# movie_id = 457
# result = predict_rating_with_pearson_correlation(user_id, movie_id, train_matrix, test_map)
# print(result)

def cosine_similarity(v1, v2):
    '''
    calculate the cosine similarity
    :param v1: python list
    :param v2: python list
    :return: int
    '''
    if len(v1) != len(v2) or len(v1) == 0 or len(v2) == 0:
        print("Error: invalid input, the length of vectors should be equal and larger than 0")
        return

    result = 0.0
    numerator = 0.0
    denominator = 0.0

    numerator = np.dot(v1, v2)
    denominator = np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2))

    if denominator != 0.0:
        result = numerator / denominator

    return result

# v1 = [1,0,1]
# v2 = [-1,0,-1]
# res = cosine_similarity(v1, v2)
# print(res)
#
# # DON'T WORK!!!
# if res == 1.0:
#     print('y')
# else: print('n')

def cal_adjusted_cosine_similarity(target_id, neighbor_id, t_train_matrix, train_mean_rate_map):
    '''
    calculate the adjusted cosine similarity in item-based approach
    :param target_id: int, the given movie id
    :param neighbor_id: the certain movie id in the train data
    :param t_train_matrix: the transposed train matrix
    :param train_mean_rate_map: mean rate of each user in the train data
    :return: float
    '''
    adj_cosine_sim = 0.0

    target_row = t_train_matrix[target_id - 1]
    neighbor_row = t_train_matrix[neighbor_id - 1]

    # filter the common component
    # subtract the mean rate
    target_vector = []
    neighbor_vector = []
    for i in range(len(t_train_matrix[0])):
        if target_row[i] > 0 and neighbor_row[i] > 0:
            target_vector.append(target_row[i] - train_mean_rate_map[i + 1])
            neighbor_vector.append((neighbor_row[i]) - train_mean_rate_map[i + 1])

    # calculate the adjusted cosine similarity
    if len(target_vector) == len(neighbor_vector) and len(target_vector) != 0:
        adj_cosine_sim = cosine_similarity(target_vector, neighbor_vector)

    return adj_cosine_sim

# num_of_users = 200
# num_of_movies = 1000
# train_matrix = [[0] * num_of_movies] * num_of_users
# build_train_matrix("train.txt", train_matrix)
# build_train_matrix("train.txt", train_matrix)
# train_mean_rate_map = avg_movie_rate_of_train_users(train_matrix)
# t_train_matrix = np.array(train_matrix).T
# target_id = 1
# neighbor_id = 1000
# res = cal_adjusted_cosine_similarity(target_id, neighbor_id, t_train_matrix, train_mean_rate_map)
# print(res)

def build_map_similar_neighbor_adj_cosine(train_matrix, train_mean_rate_map):
    '''
    build map of the similar neighbors based on adjusted cosine similarity
    :param movie_id: int
    :param train_matrix: python 2-d array
    :param train_mean_rate_map: python dictionary, K: train user id, V: mean rate of given train user
    :return: python dictionary, K: movie id, V: python dictionary(K: movie id, V: similarity)
    '''

    neighbor_map = {}

    # transpose the train matrix
    t_train_matrix = np.array(train_matrix).T

    for i in range(len(t_train_matrix)):
        target_id = i + 1

        if target_id not in neighbor_map:
            neighbor_map[target_id] = {}

        for j in range(i + 1, len(t_train_matrix)):
            neighbor_id = j + 1

            adj_cosine_sim = cal_adjusted_cosine_similarity(target_id, neighbor_id, t_train_matrix, train_mean_rate_map)

            neighbor_map[target_id][neighbor_id] = adj_cosine_sim

            if neighbor_id not in neighbor_map:
                neighbor_map[neighbor_id] = {}

            neighbor_map[neighbor_id][target_id] = adj_cosine_sim

    return neighbor_map

# num_of_users = 200
# num_of_movies = 1000
# train_matrix = [[0] * num_of_movies] * num_of_users
# build_train_matrix("train.txt", train_matrix)
# build_train_matrix("train.txt", train_matrix)
# train_mean_rate_map = avg_movie_rate_of_train_users(train_matrix)
# neighbor_map = build_map_similar_neighbor_adj_cosine(train_matrix, train_mean_rate_map)
# # print(neighbor_map)



def predict_rating_with_item_based_adj_cosine(user_id, movie_id, test_map, neighbor_map):
    '''
    predict the rate with item based adjusted cosine similarity
    :param user_id: int, test user id
    :param movie_id: int, test user's unrated movie id
    :param train_matrix: python 2-d list
    :param train_mean_rate_map: python dictionary
    :param neighbor_map: python dictionary of dictionary
    :return:
    '''
    result = 0.0
    numerator = 0.0
    denominator = 0.0

    user = test_map[user_id]
    list_of_rated_movie = user.get_list_of_rated_movie()
    list_of_rate_of_rated_movie = user.get_list_of_rate_of_rated_movie()

    # given test user's mean movie rate
    test_user_mean_rate = avg_movie_rate_of_test_user(user_id, test_map)

    # get the unrated movie's neighbor map
    map_of_neighbors = neighbor_map[movie_id]

    for i in range(len(list_of_rated_movie)):
        rated_movie_id = list_of_rated_movie[i]
        rated_movie_rate = list_of_rate_of_rated_movie[i]

        adj_cosine_sim = map_of_neighbors[rated_movie_id]
        numerator += rated_movie_rate * adj_cosine_sim
        denominator += abs(adj_cosine_sim)

    if denominator != 0.0:
        result = numerator / denominator
    else:
        result = test_user_mean_rate

    result = int(round(result))

    # the some original result is less than 1
    if result > 5:
        result = 5
    if result < 1:
        result = 1

    return result

def run(io_file, train_matrix, train_mean_rate_map):
    '''
    process input file
    :param io_file: python tuple(output file, input file)
    :param train_matrix:
    :param train_mean_rate_map:
    :return:
    '''
    out_file = open(io_file[0], "w")
    print("***")
    print("create a output file: {0}".format(io_file[0]))

    test_map = build_test_user_map(io_file[1])
    num_of_neighbor = 100

    # iuf
    # update_data_iuf(train_matrix, test_map)

    # build neighbor map based on adjusted cosine similarity
    # adj_cosine_map_of_neighbors = build_map_similar_neighbor_adj_cosine(train_matrix, train_mean_rate_map)
    # print("finish building the neighbor map")

    list_of_test_user_id = sorted(test_map.keys())
    print("start writing file: {0}".format(io_file[0]))


    for user_id in list_of_test_user_id:
        user = test_map[user_id]
        list_of_unrated_movie = user.get_list_of_unrated_movie()

        # neighbor searching based on cosine similarity
        cosine_list_of_neighbors = find_similar_neighbor_cosine(user_id, train_matrix, test_map)

        # neighbor searching based on pearson correlation
        pearson_list_of_neighbors = find_similar_neighbor_pearson(user_id, train_matrix, test_map, train_mean_rate_map)

        for movie_id in list_of_unrated_movie:
            # the predicted rating based on cosine similarity
            cosine_rating = predict_rating_with_cosine_similarity(user_id, movie_id, num_of_neighbor, train_matrix, test_map, cosine_list_of_neighbors)

            # the predicted rating based on pearson correlation
            pearson_rating = predict_rating_with_pearson_correlation(user_id, movie_id, train_matrix, test_map, train_mean_rate_map, pearson_list_of_neighbors)

            # optimized rating
            optimized_rating = int(round(0.5 * cosine_rating + 0.5 * pearson_rating))

            # the predicted rating based on item based adjusted cosine similarity
            # item_based_adj_cosine_rating = predict_rating_with_item_based_adj_cosine(user_id, movie_id, test_map, adj_cosine_map_of_neighbors)

            out_line = str(user_id) + " " + str(movie_id) + " " + str(optimized_rating) + "\n"
            out_file.write(out_line)

            # print(("user id: {0}, movie_id: {1}, rate {2}").format(user_id, movie_id, item_based_adj_cosine_rating))

    out_file.close()

    print("finish writing {0}".format(io_file[0]))
    print("***")
    print(" ")

def main():
    '''
    the main entry the program
    :return: void
    '''
    # io_list = [("./result_set/result5.txt", "test5.txt"), ("./result_set/result10.txt", "test10.txt"), ("./result_set/result20.txt", "test20.txt")]
    # train_file = "train.txt"

    # cross-validation
    io_list = [("./mae/eval_result.txt", "./mae/eval_test.txt")]
    train_file = "./mae/eval_train.txt"

    # build the train matrix from train.txt
    num_of_users = 200
    num_of_movies = 1000
    train_matrix = [[0] * num_of_movies] * num_of_users
    build_train_matrix(train_file, train_matrix)

    train_mean_rate_map = avg_movie_rate_of_train_users(train_matrix)

    for io_file in io_list:
        run(io_file, train_matrix, train_mean_rate_map)

main()