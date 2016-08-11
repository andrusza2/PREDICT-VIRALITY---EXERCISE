function basic_statistics (n, data)
  mean_val = mean(data(:,n));
  median_val = median(data(:,n));
  std_val = std(data(:,n));
  range_val = range(data(:,n));
  kurtosis_val = kurtosis(data(:,n));
  
  printf("%d:\n", n);
  printf("\tMean Value: %f\n", mean_val);
  printf("\tMedian Value: %f\n", median_val);
  printf("\tStandard deviation: %f\n", std_val);
  printf("\tRange: %f\n", range_val);
  printf("\tKurtosis: %f\n", kurtosis_val);
endfunction


