disp('-------------------------------')
disp('Testing Sparse GP vs Full GP...')
disp('-------------------------------')
clear all, close all;
%sd = 3; rand('seed',sd), randn('seed',sd)       % set a seed for reproducability

%% Basic parameters
MAX_NUM_EVAL = 100;         % Maximum allowed function evals
n_train = 1000;             % Number of training points
n_train_sparse = 100;       % Number of inducing inputs / size of active set
n_test = 5000;              % Number of test points
n_dim = 10;                  % Size of UF1 problem
n_responses = 2 ;           % Number of responses for UF1 problem
%sn = 0.001;                   % Noise standard deviation. NOT INCLUDING NOISE for now (CHECK THIS OUT!!)

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
%hyp.lik = log(sn); 
hyp.lik = logtheta0(end); 
inf = @infGaussLik;

%% Optimise hyperparameters
% fprintf('Optimising hyperparameters...\n')
% tic;
% hyp = minimize(hyp,@gp,-MAX_NUM_EVAL,inf,mean,cov,lik,X_train,y_train);      % optimise hyperparameters. param -N: gives the maximum allowed function evals
% hyperparam_full_time = toc;

%% Full GP
% [ymu,ys2] = gp(hyp,inf,mean,cov,lik,X_train,y_train,X_test);                 % dense prediction
% diff_full = compute_RMSE(ymu,y_test);

%% Sparse GPs - initial basic settings
xu = [rand(n_train_sparse,1), rand(n_train_sparse,n_dim-1)*2-1];       % inducing points randomly (n_dimensions, in UF1 design range)
cov = {'apxSparse',cov,xu};                 % change covariance function to use sparse methods

%% 1) Sparse GPs with randomly inducing points and hyper-parameters from
%     full GP, 4 methods altogether
% infv  = @(varargin) inf(varargin{:},struct('s',0.0));           % VFE, opt.s = 0
% [ymuv,ys2v] = gp(hyp,infv,mean,cov,lik,X_train,y_train,X_test);
% infs = @(varargin) inf(varargin{:},struct('s',0.2));            % SPEP, 0<opt.s<1
% [ymus,ys2s] = gp(hyp,infs,mean,cov,lik,X_train,y_train,X_test);
% inff = @(varargin) inf(varargin{:},struct('s',1.0));            % FITC, opt.s = 1
% [ymuf,ys2f] = gp(hyp,inff,mean,cov,lik,X_train,y_train,X_test);
% infe = @infFITC_EP; 
% [ymue,ys2e] = gp(hyp,infe,mean,cov,lik,X_train,y_train,X_test);
% 
% diff_sparse_vfe = compute_RMSE(ymuv,y_test);
% diff_sparse_spep = compute_RMSE(ymus,y_test);
% diff_sparse_fitc = compute_RMSE(ymuf,y_test);
% diff_sparse_fitc_ep = compute_RMSE(ymue,y_test);

%% 2) Unknown: Not sure what is the effect of this
hyp.xu = xu;
% [ymu_unknown,ys2_unknown] = gp(hyp,inf,mean,cov,lik,X_train,y_train,X_test);
% diff_sparse_unknown = compute_RMSE(ymu_unknown,y_test);

%% 3) Sparse GPs with inducing points and hyper-parameters optimised
%     from the sparse set
tic;
hyp = minimize(hyp,@gp,-MAX_NUM_EVAL,inf,mean,cov,lik,X_train,y_train);  % exactly the same as above, except cov is different
hyperparam_sparse_time = toc;

[ymu_spgp,ys2_spgp] = gp(hyp,inf,mean,cov,lik,X_train,y_train,X_test);
diff_sparse_spgp = compute_RMSE(ymu_spgp,y_test);

%% Exploring results
fprintf('Validation results performed on %d test points...\n', n_test)
% fprintf('RMSE for full GP: %f\n', diff_full)
% fprintf('RMSE for sparse VFE: %f\n', diff_sparse_vfe)
% fprintf('RMSE for sparse SPEP: %f\n', diff_sparse_spep)
% fprintf('RMSE for sparse FITC: %f\n', diff_sparse_fitc)
% fprintf('RMSE for sparse FITC_EP: %f\n', diff_sparse_fitc_ep)
% fprintf('RMSE for sparse unknown: %f\n', diff_sparse_unknown)
fprintf('RMSE for sparse SPGP: %f\n', diff_sparse_spgp)
fprintf('\n')
% fprintf('Time taken to optimise hyperparameters for full GP: %fs\n', hyperparam_full_time)
fprintf('Time taken to optimise hyperparameters for sparse GP: %fs\n', hyperparam_sparse_time)

