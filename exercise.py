import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

### Ad. (1)
### Read data from csv to DataFrame
data = pd.read_csv("data.csv", header=None)
# print data.shape

v168 = data[168].values

### Ad. (2)
### Plot distribution of v168
plt.hist(v168, bins=50)
plt.title("distribution of the v(168)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

### Ad. (3)
### Plot distribution of log(v(168))
v168_log = np.log(v168)
plt.hist(v168_log, bins=50)
plt.title("distribution of the log(v(168))")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
### Yes, it looks more "gaussian"...

### Ad. (3)
### Mean value and standard deviation
mean = v168_log.mean()
std_dev = v168_log.std()

print "Mean: ", mean
print "Standard deviation: ", std_dev

### 3-sigma values:
min_val = mean - 3 * std_dev
max_val = mean + 3 * std_dev

print "3 sigma values: ", [min_val, max_val]

### Normalize data (3-sigma rule)
normalized_data = data[(np.log(data[168]) > min_val) & (np.log(data[168]) < max_val)]
# print normalized_data.shape

v168_norm = np.log(normalized_data[168].values)

### Plot distribution of normalized log(v(168))
plt.hist(v168_norm, bins=50)
plt.title("distribution of the normalized log(v(168))")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

### Ad. (5)
### Compute correlation coefficients
for i in xrange(1, 25):
    print "n=", i
    print np.corrcoef(np.log(normalized_data[i]), v168_norm)

### Ad. (6)
### Split dataset, 10% for testing
train, test = train_test_split(normalized_data, test_size=0.1, random_state=42)

# print train.shape
# print test.shape

### Ad. (7), (8), (9)
### SKlearn linear regressor init
linear_regressor = LinearRegression()

### mRSE results variables
results_mRSE = []
results_mRSE_2 = []

### Train linear and multiple-input linear regression models, compute mRSE for n in <1, 24>
for i in xrange(1, 25):
    X = train[i].values.reshape(-1, 1)
    X2 = train.loc[:,1:i+1].values
    Y = train[168].values

    X_t = test[i].values.reshape(-1,1)
    X_t2 = test.loc[:, 1:i+1].values
    Y_t = test[168].values

    linear_regressor.fit(X, Y)
    results_mRSE.append(mean_squared_error(Y_t, linear_regressor.predict(X_t)))

    linear_regressor.fit(X2, Y)
    results_mRSE_2.append(mean_squared_error(Y_t, linear_regressor.predict(X_t2)))

# print results_mRSE


### Ad. (10)
### Plot the mRSE values computed on the test dataset
linear_line, = plt.plot(results_mRSE, label="Linear Regression")
multiple_linear_line, = plt.plot(results_mRSE_2, label="Multiple-input Linear Regression")
linear_legend = plt.legend(handles=[linear_line, multiple_linear_line])
plt.xlabel("Reference time (n)")
plt.ylabel("mRSE")
plt.show()
