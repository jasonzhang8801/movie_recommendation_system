from pandas import Series
import math
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

# correct
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

# correct
def build_train_data_frame(train_file):
    '''
    load the training data
    :param train_file: txt file
    :return: pandas data frame
    '''
    df = pd.read_table(train_file, header=None)
    return df

# correct
def avg_movie_rate_of_test_user(user_id, test_map):
    '''

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

# Cosine Similarity
def find_similar_neighbor_cosine(user_id, train_df, test_map):
    '''
    find the top k neighbors with cosine similarity
    :param user_id: int
    :param train_df: pandas DataFrame
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
    for row in range(train_df.shape[0]):
        train_user_id = row + 1
        common_movie = 0

        numerator = 0.0
        denominator = 0.0
        cosine_similarity = 0.0

        sqr_sum_of_test_rate = 0.0
        sqr_sum_of_train_rate = 0.0

        for i in range(len(list_of_rated_movie)):
            movie_id = list_of_rated_movie[i]
            test_movie_rate = list_of_rate_of_rated_movie[i]
            train_movie_rate = train_df[movie_id - 1][train_user_id - 1]

            # movie rate with zero means the user doesn't rate the movie
            # we should not consider it as part of the calculation of cosine similarity
            if train_movie_rate != 0:
                common_movie += 1

                numerator += test_movie_rate * train_movie_rate
                sqr_sum_of_test_rate += test_movie_rate ** 2
                sqr_sum_of_train_rate += train_movie_rate ** 2

        denominator = math.sqrt(sqr_sum_of_test_rate) * math.sqrt(sqr_sum_of_train_rate)

        # the common movie between test data and train data should be larger than 1
        if common_movie > 1 and denominator != 0.0:
            cosine_similarity = numerator / denominator
            list_of_neighbor.append((train_user_id, cosine_similarity))

    list_of_neighbor.sort(key=lambda tup : tup[1], reverse=True)

    return list_of_neighbor

def predict_rating_with_cosine_similarity(user_id, movie_id, num_of_neighbor, train_df, test_map):
    '''
    predict the user's rating on the given movie based on cosine similarity
    :param user_id: int
    :param movie_id: int
    :param num_of_neighbor: int
    :return: int
    '''
    list_of_neighbor = find_similar_neighbor_cosine(user_id, train_df, test_map)
    # average rate of user in the test data
    avg_movie_rate_in_test = avg_movie_rate_of_test_user(user_id, test_map)

    numerator = 0.0
    denominator = 0.0

    counter = 0
    for i in range(len(list_of_neighbor)):
        if counter > num_of_neighbor: break

        neighbor_id = list_of_neighbor[i][0]
        neighbor_similarity = list_of_neighbor[i][1]
        neighbor_movie_rate = train_df[movie_id - 1][neighbor_id - 1]

        if neighbor_movie_rate > 0:
            counter += 1
            numerator += neighbor_similarity * neighbor_movie_rate
            denominator += neighbor_similarity

        # more

    if denominator != 0.0:
        result = numerator / denominator
    else:
        result = avg_movie_rate_in_test

    result = int(round(result))

    return result



# Pearson Correlation
def cal_pearson_correlation(test_users, train_users):
    '''
    calculate the pearson correlation
    :param test_users: list of tuple(movie id, movie rate) from unrated movie in test data
    :param train_users: pandas series from pandas data frame in train data
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

    mean_test_rates = sum(vector_test_rates) / len(vector_test_rates)
    mean_train_rates = sum(vector_train_rates) / len(vector_train_rates)

    adj_vector_test_users = [movie_rate - mean_test_rates for movie_rate in vector_test_rates]
    adj_vector_train_users = [movie_rate - mean_train_rates for movie_rate in vector_train_rates]

    s_adj_vector_test_users = Series(adj_vector_test_users)
    s_adj_vector_train_users = Series(adj_vector_train_users)

    numerator = Series.dot(s_adj_vector_test_users, s_adj_vector_train_users)
    denominator = math.sqrt(Series.dot(s_adj_vector_test_users, s_adj_vector_test_users)) * \
        math.sqrt(Series.dot(s_adj_vector_train_users, s_adj_vector_train_users))

    # # no common component
    # # use the avg of all movie of test user or all movie of all test users
    # if len(adj_vector_test_users) == 0 or len(adj_vector_train_users) == 0:
    #     return 0.0, 0.0
    # # there are common components
    # # use the avg of all movie of test user or all movie of all test users
    # elif len(adj_vector_test_users) != 0 and len(adj_vector_train_users) != 0:
    #     if all(v == 0 for v in adj_vector_test_users) or all(v == 0 for v in adj_vector_train_users):
    #         return 0.0, mean_test_rates
    # return numerator / denominator, mean_test_rates

    # each component of one or both vectors is the same
    if denominator == 0.0:
        return 0.0

    return numerator / denominator

# train_df = build_train_data_frame("train.txt")
# test_users = [(42, 4), (85, 2), (194, 5), (208, 5), (585, 1)]
# train_users = train_df.iloc[0]
# res = cal_pearson_correlation(test_users, train_users)
# print(res)

def find_similar_neighbor_pearson(user_id, train_df, test_map):
    '''

    :param user_id: int
    :param train_df: pandas dataframe
    :param test_map: python dictionary
    :return: a list of tuple(user id, similarity)
    '''
    list_of_neighbors = []

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
    for row in range(train_df.shape[0]):
        train_user_id = row + 1

        # retrieve the user row as pandas series from train data frame
        train_user_series = train_df.iloc[train_user_id - 1]
        pearson_correlation = cal_pearson_correlation(zipped_list_of_rated_movie_with_rate, train_user_series)

        # correct the pearson correlation
        # range is [-1, 1]
        if pearson_correlation > 1.0:
            pearson_correlation = 1.0

        # if pearson_correlation != 0.0:
        #     list_of_neighbors.append((train_user_id, pearson_correlation))


        list_of_neighbors.append((train_user_id, pearson_correlation))

    list_of_neighbors.sort(key=lambda tup: tup[1], reverse=True)

    return list_of_neighbors

# test_map = build_test_user_map("test5.txt")
# train_df = build_train_data_frame("train.txt")
# list_of_neighbor = find_similar_neighbor_pearson(206, train_df, test_map)
# print(list_of_neighbor)

def predict_rating_with_pearson_correlation(user_id, movie_id, train_df, test_map):
    '''

    :param user_id:
    :param movie_id:
    :param train_df:
    :param test_map:
    :return:
    '''
    list_of_neighbors = find_similar_neighbor_pearson(user_id, train_df, test_map)

    # mean rate of test user
    test_user = test_map[user_id]
    list_of_rate_of_rated_movie = test_user.get_list_of_rate_of_rated_movie()
    mean_rate_of_test_user = sum(list_of_rate_of_rated_movie) / len(list_of_rate_of_rated_movie)

    numerator = 0.0
    denominator = 0.0

    for i in range(len(list_of_neighbors)):
        neighbor_id = list_of_neighbors[i][0]
        pearson_correlation_of_neighbor = list_of_neighbors[i][1]

        # mean rate of train user
        neighbor_series = train_df.iloc[neighbor_id - 1]
        neighbor_series_without_zero = neighbor_series[neighbor_series != 0]
        mean_rate_of_train_user = neighbor_series_without_zero.mean()

        train_user_rate = train_df[movie_id - 1][neighbor_id - 1]

        if train_user_rate > 0:
            numerator += pearson_correlation_of_neighbor * (train_user_rate - mean_rate_of_train_user)
            denominator += abs(pearson_correlation_of_neighbor)

    if denominator == 0.0:
        result = mean_rate_of_test_user
    else:
        result = mean_rate_of_test_user + numerator / denominator

    result = int(round(result))

    if result > 5:
        result = 5

    return result

# test_map = build_test_user_map("test5.txt")
# train_df = build_train_data_frame("train.txt")
# result = predict_rating_with_pearson_correlation(206, 672, train_df, test_map)
# print(result)

def run(io_file):
    '''
    the main function of the program
    :return: txt file
    '''
    out_file = open(io_file[0], "w")
    print("successfully create a output file")

    test_map = build_test_user_map(io_file[1])
    train_df = build_train_data_frame("train.txt")
    num_of_neighbor = 100

    list_of_test_user_id = sorted(test_map.keys())
    print("start to write {0}".format(io_file[0]))

    for user_id in list_of_test_user_id:
        user = test_map[user_id]
        list_of_unrated_movie = user.get_list_of_unrated_movie()

        for movie_id in list_of_unrated_movie:
            # the predicted rating based on cosine similarity
            rating = predict_rating_with_cosine_similarity(user_id, movie_id, num_of_neighbor, train_df, test_map)

            # the predicted rating based on pearson correlation
            # rating = predict_rating_with_pearson_correlation(user_id, movie_id, train_df, test_map)

            out_line = str(user_id) + " " + str(movie_id) + " " + str(rating) + "\n"
            out_file.write(out_line)
            # print(("wrote user id is {0} movie_id is {1} rating is {2}").format(user_id, movie_id, rating))

    out_file.close()
    print("finish writing {0}".format(io_file[0]))

def main():
    '''

    :return:
    '''
    io_list = [("result5.txt", "test5.txt"), ("result10.txt", "test10.txt"), ("result20.txt", "test20.txt")]

    for io_file in io_list:
        run(io_file)

main()