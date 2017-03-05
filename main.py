import pandas as pd
import math

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


def find_similar_neighbor(user_id, train_df, test_map):
    '''
    find the top k neighbors with cosine similarity
    :param user_id: int
    :param train_df: pandas DataFrame
    :param test_map: object TestUserMap
    :return: list_of_tuple(user_id, cosine_similarity)
    '''
    test_user = test_map.get(user_id)
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
    list_of_neighbor = find_similar_neighbor(user_id, train_df, test_map)
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

def main():
    '''
    the main function of the program
    :return: txt file
    '''
    out_file = open("result20.txt", "w")
    print("successfully create a output file")

    test_map = build_test_user_map("test20.txt")
    train_df = build_train_data_frame("train.txt")
    num_of_neighbor = 100

    list_of_test_user_id = sorted(test_map.keys())
    print("start to write...")

    for user_id in list_of_test_user_id:
        user = test_map[user_id]
        list_of_unrated_movie = user.get_list_of_unrated_movie()

        for movie_id in list_of_unrated_movie:
            # the predicted rating based on cosine similarity
            rating = predict_rating_with_cosine_similarity(user_id, movie_id, num_of_neighbor, train_df, test_map)

            out_line = str(user_id) + " " + str(movie_id) + " " + str(rating) + "\n"
            out_file.write(out_line)
            print(("wrote user id is {0} movie_id is {1} rating is {2}").format(user_id, movie_id, rating))

    out_file.close()

main()
