function ret = train_multiple_linear_model_and_compute_mRSE(i, train, test)
  [beta, sigma, r] = ols(train(:,168), train(:,1:i));
  
  prediction = test(:, 1:i)*beta;
  real = test(:,168);

  ret = mean((prediction./real - 1).^2);
end
