function ret = correlation(v1, v2)
  ret = corr(log(v1+1), log(v2+1));
endfunction