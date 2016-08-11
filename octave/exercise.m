% Read data
data = dlmread('data.csv', ",", 0, 1);

% Set rand seed
rand("seed", 12);

% Print basic statistics
for i = [24,72,168]
  basic_statistics(i, data);
end

v168 = data(:,168);

% Plot distributions
plot_distribution(v168);
plot_distribution(log(v168));

% 3-sigma (removing outsiders)
normalized_data = three_sigma_normalization(data);

% Correlation coefficients
for i = 1:24
  printf("Correlation coefficient for n=%d: %f\n", i, correlation(data(:,i), v168));
end

% Split data for training and testing
n_rows = rows(normalized_data);
rand_rows = randperm(n_rows);

train = normalized_data(rand_rows(1:round(0.9*n_rows)),:);
test = normalized_data(rand_rows(round(0.9*n_rows)+1:end),:);

% Train linear models
linear_mRSE = [];
multiple_mRSE = [];

for i=(1:24)
  linear_mRSE(i) = train_single_linear_model_and_compute_mRSE(i, train, test);
  multiple_mRSE(i) = train_multiple_linear_model_and_compute_mRSE(i, train, test);
end

% Final plot
figure; % open a new figure window
  
plot((1:24), linear_mRSE, '-', (1:24), multiple_mRSE, '-');




