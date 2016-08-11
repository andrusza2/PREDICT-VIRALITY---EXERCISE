function ret = three_sigma_normalization(data)
  v168 = data(:,168);
  
  v168_mean = mean(log(v168));
  v168_std = std(log(v168));
  
  min_val = v168_mean - 3*v168_std;
  max_val = v168_mean + 3*v168_std;

  ret = data((log(data(:,168)) > min_val) & (log(data(:,168)) < max_val), :);
endfunction