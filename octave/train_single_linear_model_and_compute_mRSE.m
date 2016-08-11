function ret = train_single_linear_model_and_compute_mRSE(i, train, test)
  [beta, sigma, r] = ols(train(:,168), train(:,i));
  
  prediction = test(:,i) * beta;
  real = test(:,168);

  ret = mean((prediction./real - 1).^2);
end