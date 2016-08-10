import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def print_basic_statistics(n, data):
    print " n = ", n
    v = data[n].values
    print "     Mean value = ", v.mean()
    print "     Standard deviation = ", v.std()


def plot_distribution(vector, name):
    plt.hist(vector, bins=50)
    plt.title("distribution of the " + name)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


def three_sigma_normlaization(data):
    v168 = data[168].values
    v168_log = np.log(v168 + 1)

    ### Mean value and standard deviation
    mean = v168_log.mean()
    std_dev = v168_log.std()

    print "Removing outsiders:"
    print " Mean value: ", mean
    print " Standard deviation: ", std_dev

    ### 3-sigma values:
    min_val = mean - 3 * std_dev
    max_val = mean + 3 * std_dev

    print " 3-sigma values: ", [min_val, max_val]

    ### Normalize data (3-sigma rule)
    normalized_data = data[(np.log(data[168] + 1) > min_val) & (np.log(data[168] + 1) < max_val)]
    return normalized_data


def plot_mRSE_values(linear_results, multiple_linear_results):
    ax = plt.subplot()
    linear_line, = ax.plot(np.arange(1, 25, 1), linear_results, label="Linear Regression")
    multiple_linear_line, = ax.plot(np.arange(1, 25, 1), multiple_linear_results, label="Multiple-input Linear Regression")
    linear_legend = plt.legend(handles=[linear_line, multiple_linear_line])
    plt.xlabel("Reference time (n)")
    plt.ylabel("mRSE")
    plt.show()


def train_linear_model_and_compute_mRSE(i, train, test):

    ## Log transformed
    X = np.log(train[i].values + 1).reshape(-1, 1)
    X2 = np.log(train.loc[:, 1:i].values + 1)
    Y = np.log(train[168].values + 1)

    X_t = np.log(test[i].values + 1).reshape(-1, 1)
    X_t2 = np.log(test.loc[:, 1:i].values + 1)
    Y_t = np.log(test[168].values + 1)

    # ## Raw views
    # X = train[i].values.reshape(-1, 1)
    # X2 = train.loc[:, 1:i].values
    # Y = train[168].values
    #
    # X_t = test[i].values.reshape(-1, 1)
    # X_t2 = test.loc[:, 1:i].values
    # Y_t = test[168].values

    linear_regressor.fit(X, Y)
    single_mRSE = mean_squared_error(Y_t, linear_regressor.predict(X_t))

    linear_regressor.fit(X2, Y)
    multiple_mRSE = mean_squared_error(Y_t, linear_regressor.predict(X_t2))

    return single_mRSE, multiple_mRSE


if __name__ == "__main__":
    ### Ad. (1)
    ### Read data from csv to DataFrame
    data = pd.read_csv("data.csv", header=None)

    v168 = data[168].values

    print "Basic Statistics:"
    for n in [24, 72, 168]:
        print_basic_statistics(n, data)


    ### Ad. (2)
    ### Plot distribution of v168
    plot_distribution(v168, "v(168)")
    ### It's irregular distribution...


    ### Ad. (3)
    ### Plot distribution of log(v(168))
    v168_log = np.log(v168+1)
    plot_distribution(v168_log, "log(v(168)+1)")
    ### Yes, it looks more "gaussian"...


    ### Ad. (4)
    ### Normalize data (3-sigma rule)
    normalized_data = three_sigma_normlaization(data)

    v168_norm = np.log(normalized_data[168].values + 1)

    ### Plot distribution of normalized log(v(168))
    plot_distribution(v168_norm, "normalized log(v(168))")


    ### Ad. (5)
    ### Compute correlation coefficients
    print "Correlation coefficients:"
    for i in xrange(1, 25):
        print " n = ", i, ": ", np.corrcoef(np.log(normalized_data[i] + 1), v168_norm)[0][1]


    ### Ad. (6)
    ### Split dataset, 10% for testing
    train, test = train_test_split(normalized_data, test_size=0.1, random_state=1)


    ### Ad. (7), (8), (9)
    ### SKlearn linear regressor init
    linear_regressor = LinearRegression()

    ### mRSE results variables
    results_mRSE = []
    results_mRSE_2 = []

    ### Train linear and multiple-input linear regression models, compute mRSE for n in <1, 24>
    for i in xrange(1, 25):
        single_mRSE, multiple_mRSE = train_linear_model_and_compute_mRSE(i, train, test)
        results_mRSE.append(single_mRSE)
        results_mRSE_2.append(multiple_mRSE)


    ### Ad. (10)
    ### Plot the mRSE values computed on the test dataset
    plot_mRSE_values(results_mRSE, results_mRSE_2)
