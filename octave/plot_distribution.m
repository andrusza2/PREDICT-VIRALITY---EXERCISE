function plot_distribution (vector_data)
  figure; % open a new figure window
  
  hist(vector_data, 30);
  title('Distribution');
  xlabel('Value');
  ylabel('Frequency')
endfunction