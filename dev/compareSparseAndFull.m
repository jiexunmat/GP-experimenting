disp('-------------------------------')
disp('Full GP vs Sparse GP...')
disp('-------------------------------')
clear all, close all;
%sd = 3; rand('seed',sd), randn('seed',sd)       % set a seed for reproducability

%% Basic parameters
MAX_NUM_EVAL_FULL = 50;         % Maximum allowed function evals for full GP
MAX_NUM_EVAL_SPARSE = 200;      % Maximum allowed function evals for sparse GP
n_train = 800;                 % Number of training points
n_train_sparse = n_train/10;    % Number of inducing inputs / size of active set
n_test = 5000;                  % Number of test points
n_dim = 3;                      % Size of UF1 problem
n_responses = 2 ;               % Number of responses for UF1 problem
%sn = 0.001;                    % Noise standard deviation. NOT INCLUDING NOISE for now (CHECK THIS OUT!!)


n_trials = 20;
useOld = 1;

%% Initialise trackers
diff_old = zeros(1, n_trials);
diff_full = zeros(1, n_trials);
diff_sparse = zeros(1, n_trials);
hyperparam_old_time = zeros(1, n_trials);
hyperparam_full_time = zeros(1, n_trials);
hyperparam_sparse_time = zeros(1, n_trials);

for j = 1:n_trials
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
    hyp=[];inf=[];lik=[];cov=[];
    cov = {@covSum, {@covSEard,@covNoise}}; 
    hyp.cov = logtheta0;  
    emptymean = [];
    lik = {@likGauss};    
    hyp.lik = logtheta0(end);
    inf = @infGaussLik;
    %inf  = @(varargin) inf(varargin{:},struct('s',0.2));   

    %% Use old GP method
    if useOld
        gpoptions.covfunc = {'covSum', {'covSEard','covNoise'}};
        gpoptions.logtheta0 = logtheta0;
        fprintf('Optimising hyperparameters for old GP...\n')
        tic;
        gpdata = gaussianprocessregression('Train', X_train, y_train, gpoptions);
        hyperparam_old_time(j) = toc;

        % Generate predictions
        ymu_old = gaussianprocessregression('Evaluate', X_test, gpdata);
        diff_old(j) = compute_RMSE(ymu_old,y_test);
    end
    %% Full GP
    % Optimise hyperparameters
    fprintf('Optimising hyperparameters for full GP...\n')
    tic;
    hyp = minimize_minfunc(hyp,@gp,-MAX_NUM_EVAL_FULL,inf,emptymean,cov,lik,X_train,y_train);      
    hyperparam_full_time(j) = toc;

    % Generate predictions
    [ymu,ys2] = gp(hyp,inf,emptymean,cov,lik,X_train,y_train,X_test);                 % dense prediction
    diff_full(j) = compute_RMSE(ymu,y_test);

    %% Reset hyperparameters for sparse GP
    % Use covariance function (from Swordfish)
    cov = {@covSum, {@covSEard,@covNoise}}; 
    hyp.cov = logtheta0;   
    hyp.lik = logtheta0(end); 

    %% Sparse GPs - initial basic settings
    % Initialise inducing points randomly from X_train
    % indices = randperm(n_train);
    % sparse_indices = indices(1:n_train_sparse);
    % xu = X_train(sparse_indices,:);
    xu = lhs(lb,ub,n_train_sparse);
    %xu = [rand(n_train_sparse,1), rand(n_train_sparse,n_dim-1)*2-1];       
    cov = {'apxSparse',cov,xu};                                            % change covariance function to use sparse methods

    % Optimise hyperparameters
    hyp.xu = xu;
    fprintf('Optimising hyperparameters for sparse GP...\n')
    tic;
    hyp = minimize_minfunc(hyp,@gp,-MAX_NUM_EVAL_SPARSE,inf,emptymean,cov,lik,X_train,y_train);  % exactly the same as above, except cov is different
    hyperparam_sparse_time(j) = toc;

    % Generate predictions
    [ymu_spgp,ys2_spgp] = gp(hyp,inf,emptymean,cov,lik,X_train,y_train,X_test);
    diff_sparse(j) = compute_RMSE(ymu_spgp,y_test);

end
%% Exploring results
fprintf('Average validation results performed on %d test points over %d trials...\n', n_test, n_trials)
if useOld
    fprintf('RMSE for old GP: %f\n', mean(diff_old))
end
fprintf('RMSE for full GP: %f\n', mean(diff_full))
fprintf('RMSE for sparse SPGP: %f\n', mean(diff_sparse))
fprintf('\n')
if useOld
   fprintf('Time taken to optimise hyperparameters for old GP: %fs\n', mean(hyperparam_old_time)) 
end
fprintf('Time taken to optimise hyperparameters for full GP: %fs\n', mean(hyperparam_full_time))
fprintf('Time taken to optimise hyperparameters for sparse GP: %fs\n', mean(hyperparam_sparse_time))

