disp('-------------------------------')
disp('Full GP vs Sparse GP...')
disp('-------------------------------')
clear all, close all;
%sd = 3; rand('seed',sd), randn('seed',sd)       % set a seed for reproducability

%% Basic parameters
MAX_NUM_EVAL_FULL = 100;        % Maximum allowed function evals for full GP
MAX_NUM_EVAL_SPARSE = 200;      % Maximum allowed function evals for sparse GP
n_train = 3000;                 % Number of training points
n_train_sparse = 500;           % Number of inducing inputs / size of active set
n_test = 5000;                  % Number of test points
n_dim = 10;                     % Size of UF1 problem
n_responses = 2 ;               % Number of responses for UF1 problem
%sn = 0.001;                    % Noise standard deviation. NOT INCLUDING NOISE for now (CHECK THIS OUT!!)

%% Setting up data - training and test
% Create training data 
lb = zeros(1,n_dim);
lb(2:end) = lb(2:end) - 1;
ub = ones(1,n_dim);
X_train = lhs(lb,ub,n_train);
% figure
% scatter3(X_train(:,1), X_train(:,2), X_train(:,3));

tmp_y = zeros(n_train,n_responses);
for i=1:n_train
    tmp_y(i,:) = UF1(X_train(i,:)')';
end
y_train = tmp_y(:,1);   % Only test on the first response, z_1

% Create test data 
X_test = [rand(n_test,1), rand(n_test,n_dim-1)*2-1];
tmp_ys = zeros(n_test,2);
for i=1:n_test
    tmp_ys(i,:) = UF1(X_test(i,:)')';
end
y_test = tmp_ys(:,1);                        

%% Setting up cov, mean, inf functions
% Initialise guess: logtheta0 (from Swordfish)
stdX = std(X_train)';
stdX( stdX./abs(mean(X_train))' < 100*eps ) = Inf; % Infinite lengthscale if range is negligible
logtheta0 = log([stdX; std(y_train); 0.05*std(y_train)]);

% Use covariance function (from Swordfish)
cov = {@covSum, {@covSEard,@covNoise}}; 
hyp.cov = logtheta0;  
mean = [];
lik = {@likGauss};    
hyp.lik = logtheta0(end); 
inf = @infGaussLik;

%% Full GP
% Optimise hyperparameters
fprintf('Optimising hyperparameters for full GP...\n')
tic;
hyp = minimize(hyp,@gp,-MAX_NUM_EVAL_FULL,inf,mean,cov,lik,X_train,y_train);      % optimise hyperparameters. param -N: gives the maximum allowed function evals
hyperparam_full_time = toc;

% Generate predictions
[ymu,ys2] = gp(hyp,inf,mean,cov,lik,X_train,y_train,X_test);                 % dense prediction
diff_full = compute_RMSE(ymu,y_test);

%% Reset hyperparameters for sparse GP
% Use covariance function (from Swordfish)
cov = {@covSum, {@covSEard,@covNoise}}; 
hyp.cov = logtheta0;  
mean = [];
lik = {@likGauss};    
hyp.lik = logtheta0(end); 
inf = @infGaussLik;

%% Sparse GPs - initial basic settings
% Initialise inducing points randomly from X_train
indices = randperm(n_train);
sparse_indices = indices(1:n_train_sparse);
xu = X_train(sparse_indices,:);
%xu = [rand(n_train_sparse,1), rand(n_train_sparse,n_dim-1)*2-1];       
cov = {'apxSparse',cov,xu};                                            % change covariance function to use sparse methods

% Optimise hyperparameters
hyp.xu = xu;
fprintf('Optimising hyperparameters for sparse GP...\n')
tic;
hyp = minimize(hyp,@gp,-MAX_NUM_EVAL_SPARSE,inf,mean,cov,lik,X_train,y_train);  % exactly the same as above, except cov is different
hyperparam_sparse_time = toc;

% Generate predictions
[ymu_spgp,ys2_spgp] = gp(hyp,inf,mean,cov,lik,X_train,y_train,X_test);
diff_sparse_spgp = compute_RMSE(ymu_spgp,y_test);

%% Exploring results
fprintf('Validation results performed on %d test points...\n', n_test)
fprintf('RMSE for full GP: %f\n', diff_full)
fprintf('RMSE for sparse SPGP: %f\n', diff_sparse_spgp)
fprintf('\n')
fprintf('Time taken to optimise hyperparameters for full GP: %fs\n', hyperparam_full_time)
fprintf('Time taken to optimise hyperparameters for sparse GP: %fs\n', hyperparam_sparse_time)

