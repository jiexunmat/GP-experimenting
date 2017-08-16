clear all;

% Initialise basic parameters 
n_samples = 100;
n_dim = 20;
ell = rand(n_dim,1); 
sf = 1;
x = rand(n_samples, n_dim);

% Using GPML covariance function
covfunc = {@covSEard}; 
hyp.cov = log([ell; sf]);
K = feval(covfunc{:}, hyp.cov, x);

% Testing homemade covariance function
hyp_test = [ell; sf];
K_test = covSEbasic(x,x,hyp_test);

% Compare differences
diff = abs(K - K_test);
total_diff = sum(sum(diff));
disp('Total discrepancy = ');
disp(total_diff);
