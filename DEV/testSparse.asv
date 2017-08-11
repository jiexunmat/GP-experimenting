disp('Testing Sparse GP vs Full GP.')
clear all, close all, write_fig = 0; N = 100;
sd = 3; rand('seed',sd), randn('seed',sd)       % set a seed for reproducability

% fprintf('a) switch between FITC/VFE/SPEP via the opt.s parameter\n')

%% Setting up data
% x = load('x');
% x = x.x;
% n_dim = size(x, 2);
% y = load('y');
% y = y.y;
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

stdX = std(x)';
stdX( stdX./abs(mean(x))' < 100*eps ) = Inf; % Infinite lengthscale if range is negligible
logtheta0 = log([stdX; std(y); 0.05*std(y)]);

cov = {@covSum, {@covSEard,@covNoise}}; hyp.cov = logtheta0; %ell = 1.0; sf = 2; noisef = 2; 
mean = [];
% mean = {@meanSum,{@meanLinear,@meanConst}}; hyp.mean = [0.5;1];
if isequal(liktyp,'g')
  lik = {@likGauss};    hyp.lik = log(sn); inf = @infGaussLik;
else
  lik = {@likLogistic}; hyp.lik = [];      inf = @infLaplace;  y = sign(y);
end

%% Optimise hyperparameters
fprintf('Optimise hyperparameters.\n')
tic;
hyp = minimize(hyp,@gp,-N,inf,mean,cov,lik,x,y);      % optimise hyperparameters. param -N: gives the maximum allowed function evals
hyper_time = toc;
fprintf('Hyperparameters optimised in %f seconds.\n', hyper_time)

%% Full GP
fprintf('Running Full GP.\n')
tic;
[ymu,ys2] = gp(hyp,inf,mean,cov,lik,x,y,xs);                  % dense prediction
% [nlZ,dnlZ] = gp(hyp,inf,mean,cov,lik,x,y); % marginal likelihood and derivatives
gp_time = toc;
fprintf('GP ran in %f seconds.\n', gp_time)

diff = sum(abs(ys - ymu));
fprintf('Discrepancy for full GP: %f\n', diff)

%% Sparse GP
nu = 50;
xu = [rand(nu,1), rand(nu,n_dim-1)*2-1];      % inducing points (n_dimensions, in UF1 design range)
cov = {'apxSparse',cov,xu};

tic;
infv  = @(varargin) inf(varargin{:},struct('s',0.0));           % VFE, opt.s = 0
[ymuv,ys2v] = gp(hyp,infv,mean,cov,lik,x,y,xs);
infs = @(varargin) inf(varargin{:},struct('s',0.7));           % SPEP, 0<opt.s<1
[ymus,ys2s] = gp(hyp,infs,mean,cov,lik,x,y,xs);
inff = @(varargin) inf(varargin{:},struct('s',1.0));           % FITC, opt.s = 1
[ymuf,ys2f] = gp(hyp,inff,mean,cov,lik,x,y,xs);

sparse_time = toc;
fprintf('Sparse GPs ran in %f seconds.\n', sparse_time)

fprintf('b) we can run sparse EP for FITC, as well\n')
infe = @infFITC_EP; [ymue,ys2e] = gp(hyp,infe,mean,cov,lik,x,y,xs);


%% Plot
subplot(211)
plot(xs,ymu,'k','LineWidth',2), hold on
plot(xs,ymuv,'g-.','LineWidth',2)
plot(xs,ymus,'m:','LineWidth',2)
plot(xs,ymuf,'c--','LineWidth',2)
plot(xs,ymue,'y:','LineWidth',2)
legend('exact','VFE','SPEP','FITC','FITC_E_P'), title('Predictive mean')
plot(x,y,'r+'), plot(xs,ys,'r')
plot(xs,ymu+2*sqrt(ys2),'k'), plot(xs,ymu-2*sqrt(ys2),'k')
xlim([-8,10]), ylim([-3,6])

subplot(212)
plot(xs,sqrt(ys2),'k','LineWidth',2), hold on
plot(xs,sqrt(ys2v),'g-.','LineWidth',2)
plot(xs,sqrt(ys2s),'m:','LineWidth',2)
plot(xs,sqrt(ys2f),'c--','LineWidth',2)
plot(xs,sqrt(ys2e),'y:','LineWidth',2)
legend('exact','VFE','SPEP','FITC','FITC_E_P'), title('Predictive standard dev')
xlim([-8,10]), if write_fig, print -depsc f11.eps; end

fprintf('c) specify inducing points via\n')
fprintf('1) hyp.xu or 2) {''apxSparse'',cov,xu}\n')
[nlZ1,dnlZ1] = gp(hyp,inf,mean,cov,lik,x,y); dnlZ1
hyp.xu = xu;
[nlZ2,dnlZ2] = gp(hyp,inf,mean,cov,lik,x,y); dnlZ2
fprintf('  The second has priority and\n')
fprintf('  results in derivatives w.r.t. xu\n')

fprintf('d) optimise nlZ w.r.t. inducing inputs\n')
fprintf('   by gradient descent\n')
hyp = minimize(hyp,@gp,-N,inf,mean,cov,lik,x,y);  % exactly the same as above, except cov is different

[ymuv,ys2v] = gp(hyp,inf,mean,cov,lik,x,y,xs);
diff_sparse = sum(abs(ys - ymuv));
fprintf('Discrepancy for sparse GP: %f\n', diff_sparse)

%% this part below gives an error
% if isequal(liktyp,'g')
%   fprintf('   and by discrete swapping\n')
%   z = 3*randn(100,1); % candidate inducing points
%   nswap = N; % number of swaps between z and hyp.xu 
%   [hyp,nlZ] = vfe_xu_opt(hyp,mean,cov,x,y,z,nswap);
% end