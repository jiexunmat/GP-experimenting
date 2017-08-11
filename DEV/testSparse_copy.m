disp('Testing Sparse GP vs Full GP...')
clear all, close all, write_fig = 0; N = 100;
sd = 3; rand('seed',sd), randn('seed',sd)       % set a seed for reproducability

%% Setting up data
sn = 0.5;                                   % noise standard deviation. NOT INCLUDING NOISE for now

% try using UF1 function to produce training points, if results not good
n = 1000;
n_dim = 3;
x = [rand(n,1), rand(n,n_dim-1)*2-1];
tmp_y = zeros(n,2);
for i=1:n
    tmp_y(i,:) = UF1(x(i,:)')';
end
y = tmp_y(:,1);

ns = 100;                                   % number of test points
xs = [rand(ns,1), rand(ns,n_dim-1)*2-1];
tmp_ys = zeros(ns,2);
for i=1:ns
    tmp_ys(i,:) = UF1(xs(i,:)')';
end
ys = tmp_ys(:,1);                           % testing only the first response

%% Setting up cov, mean, inf functions
liktyp = 'g';

% Initialise guess: logtheta0 (from Swordfish)
stdX = std(x)';
stdX( stdX./abs(mean(x))' < 100*eps ) = Inf; % Infinite lengthscale if range is negligible
logtheta0 = log([stdX; std(y); 0.05*std(y)]);

% Use covariance function (from Swordfish)
cov = {@covSum, {@covSEard,@covNoise}}; 
hyp.cov = logtheta0;  
mean = [];
lik = {@likGauss};    
hyp.lik = log(sn); 
inf = @infGaussLik;

%% Optimise hyperparameters
fprintf('Optimise hyperparameters.\n')
tic;
hyp = minimize(hyp,@gp,-N,inf,mean,cov,lik,x,y);      % optimise hyperparameters. param -N: gives the maximum allowed function evals
hyperparam_full_time = toc;

%% Full GP
fprintf('Running Full GP...\n')
[ymu,ys2] = gp(hyp,inf,mean,cov,lik,x,y,xs);                  % dense prediction
diff_full = compute_RMSE(ymu,ys);

%% Sparse GPs - initial basic settings

nu = 100;                                      % number of inducing points
xu = [rand(nu,1), rand(nu,n_dim-1)*2-1];       % inducing points randomly (n_dimensions, in UF1 design range)
cov = {'apxSparse',cov,xu};

%% 1) Sparse GPs with randomly induced points and hyper-parameters from
%     full GP, 4 methods altogether
infv  = @(varargin) inf(varargin{:},struct('s',0.0));           % VFE, opt.s = 0
[ymuv,ys2v] = gp(hyp,infv,mean,cov,lik,x,y,xs);
infs = @(varargin) inf(varargin{:},struct('s',0.2));           % SPEP, 0<opt.s<1
[ymus,ys2s] = gp(hyp,infs,mean,cov,lik,x,y,xs);
inff = @(varargin) inf(varargin{:},struct('s',1.0));           % FITC, opt.s = 1
[ymuf,ys2f] = gp(hyp,inff,mean,cov,lik,x,y,xs);
infe = @infFITC_EP; 
[ymue,ys2e] = gp(hyp,infe,mean,cov,lik,x,y,xs);

diff_sparse_vfe = compute_RMSE(ymuv,ys);
diff_sparse_spep = compute_RMSE(ymus,ys);
diff_sparse_fitc = compute_RMSE(ymuf,ys);
diff_sparse_fitc_ep = compute_RMSE(ymue,ys);

%% 2) Not sure what is the effect of this
hyp.xu = xu;
[ymu_unknown,ys2_unknown] = gp(hyp,inf,mean,cov,lik,x,y,xs);
diff_sparse_unknown = compute_RMSE(ymu_unknown,ys);

%% 3) Sparse GPs induced points and hyper-parameters optimised
%     from the sparse set
tic;
hyp = minimize(hyp,@gp,-N,inf,mean,cov,lik,x,y);  % exactly the same as above, except cov is different
hyperparam_sparse_time = toc;

[ymu_spgp,ys2_spgp] = gp(hyp,inf,mean,cov,lik,x,y,xs);
diff_sparse_spgp = compute_RMSE(ymu_spgp,ys);

% Print results
fprintf('RMSE for full GP: %f\n', diff_full)
fprintf('RMSE for sparse VFE: %f\n', diff_sparse_vfe)
fprintf('RMSE for sparse SPEP: %f\n', diff_sparse_spep)
fprintf('RMSE for sparse FITC: %f\n', diff_sparse_fitc)
fprintf('RMSE for sparse FITC_EP: %f\n', diff_sparse_unknown)
fprintf('RMSE for sparse SPGP: %f\n', diff_sparse_spgp)


