function varargout = gaussianprocessregression(Action,X,varargin)
%GAUSSIANPROCESSREGRESSION Low level gaussian process regression function.
%
%       GPDATA = GAUSSIANPROCESSREGRESSION('Train',X,Y,GPOPTIONS)
%       In X each row represents a data point and each column a
%       dimension of the inputs. The monodimensional output Y needs to be
%       arranged as a column vector. The argument GPOPTIONS contains 
%       information on how the gaussian process must be trained.
%       GPOPTIONS needs to have the following field:
%           - covfunc: Covariance functions, e.g. {'covSum', {'covSEard','covNoise'}}
%       And can optionally have the fields:
%           - logtheta0: Initial guess of the hyperparameters
%           - algorithm: Specifies the algorithm used to find the
%             hyperparameters. If 'Knitro' is selected, uses mknitro with
%             analytical derivatives.
%           - numMaxFunEvaluations: Specifies the maximum number of
%             function evaluations if using the default algorithm
%             'minimise'
%       GPDATA is a structure containing the original training set,
%       information about the covariance function, the values of the
%       hyper-parameters and other parameters needed to evaluate the
%       gaussian process.
%
%       YHAT = GAUSSIANPROCESSREGRESSION('Evaluate',X,GPDATA)
%       Returns the evaluation at X. In X each row represents a
%       data point and each column a dimension of the input.
%
%       [YHAT,S2] = GAUSSIANPROCESSREGRESSION('Evaluate',X,GPDATA)
%       Returns the variance as well. This is the gaussian process
%       assessment of its own variance and it should not be blindly
%       trusted.
%
%       See also easygpr, gker
%
%       Author(s): R. Frigola, 04-06-09
%       Copyright (c) 2009 McLaren Racing Limited.
%       $Revision: ? $  $Date: ? $ $Author: roger.frigola $  

% DEV: Increase max size
%MAX_SIZE_STORED_COVMATRIX = 1000;
MAX_SIZE_STORED_COVMATRIX = 10000;

switch Action
    case 'Train'
        y = varargin{1};
        gpoptions = varargin{2};
        dimInputs = size(X,2);
        
        % Check that there is more than one training point
        if numel(y) == 1
            progress('Close', progressId)
            errordlg('There is little to be gained by fitting a metamodel to a single point...','Only one point to fit metamodel');
            merror('Store', 0, ['Swordfish: There is only one point in the training set.']);
            merror('Raise');
        end
        
        % Demean
        offset = mean(y);
        y = y - offset;
        
        % Are we in Swordfish? 
        dbs = dbstack;
        bSwordfish = numel(dbs)>1 && strcmp(dbs(2).file,'metamodel.m');
        
        % Check if the covariance function covfunc is the Automatic
        % Relevance Determination function with noise.
        bARDNoise = numel(gpoptions.covfunc)==2 && numel(gpoptions.covfunc{2})==2 && ...
            strcmp(gpoptions.covfunc{2}{1},'covSEard') && strcmp(gpoptions.covfunc{2}{2},'covNoise');
        
        % Initial guess for the hyperparameter optimization, we make a
        % distinction depending on whether we are in Swordfish or not.
        if isfield(gpoptions,'logtheta0')
            logtheta0 = gpoptions.logtheta0(:);
        
        % DEV: always use this method, not ARD
        %elseif ~bSwordfish && bARDNoise
        elseif bARDNoise
            % Lengthscales are based on the standard deviation of the
            % inputs. We guess a signal to noise ratio of 20:1.
            stdX = std(X)';
            stdX( stdX./abs(mean(X))' < 100*eps ) = Inf; % Infinite lengthscale if range is negligible
            logtheta0 = log([stdX; std(y); 0.05*std(y)]);
            
        % DEV: don't use ARD
%         elseif bSwordfish && bARDNoise
%             % Automatic Relevance Determination
%             % http://www.gaussianprocess.org/gpml/code/matlab/doc/regression.html
%             % logtheta = [log(ell_1), ..., log(ell_dimInputs),
%             % log(sigma_f), log(sigma_n)]
%             logtheta0 = [ones(dimInputs,1); log(std(y)); log(std(y)/10)];  % Minimisation starting point
        else
            error('The field logtheta0 is necessary in GPOPTIONS for this kind of covariance function')
        end
        
        % Estimate hyperparameters
        logtheta = EstimateHyperparameters(logtheta0, gpoptions, X, y);
        
        % Very occasionally we get a bad fit: the metamodel is constant
        % everywhere except for very sharp bumps that jump directly into
        % the training points. This gives almost no training error but
        % appalling generalisation capability. Therefore, we check if some
        % lengthscales are very short. If this is the case, it is likely
        % that we have that problem and we compute the hyperparameters
        % again from a different starting point. 
        % We only do that in Swordfish where we assume that the input
        % variables in X vary between zero and one.
        if  bSwordfish && sum(logtheta(1:end-2)<-3) > 1 
            % The hyperparameters have very short lengthscales. We change
            % the initial conditions slightly and hope to find a more
            % sensible set of hyperparameters.
            % Note: I notice that problematic runs have an
            % exp(logtheta(end-1)) very similar to std(y). This could,
            % maybe, be used in the if expression above to decide whether
            % we want to re-estimate the hyperparameters or not.
            % Starting from a very noisy point seems to do the trick at the
            % moment.
            logtheta0(end) = 10;
            logtheta = EstimateHyperparameters(logtheta0, gpoptions, X, y);
        end
        
        % The problem with non-parametric methods is that we need to carry
        % all the training data!
        gpdata.x = X;
        gpdata.y = y;
        gpdata.covfunc = gpoptions.covfunc;
        gpdata.logtheta = logtheta;
        gpdata.offset = offset;
        
        % Store factored training covariance matrix
        if size(X,1) < MAX_SIZE_STORED_COVMATRIX
            K = feval(gpoptions.covfunc{:}, logtheta, X);   % compute training set covariance matrix
            try
                gpdata.L = chol(K)';                            % cholesky factorization of the covariance
            catch
                K = IncreaseNoiseLevel(gpoptions.covfunc, logtheta, X);
                gpdata.L = chol(K)';
            end
            gpdata.alpha = solve_chol(gpdata.L',y);
        end
        
        varargout{1} = gpdata;
        
    case 'Evaluate'
        gpdata = varargin{1};
        
        if isfield(gpdata,'L') % We have factored training covariance matrix
            if nargout == 2
                % Compute variance as well as mean
                [mu,varargout{2}] = gpr(gpdata.logtheta, gpdata.covfunc, gpdata.x, gpdata.y, X, gpdata.L, gpdata.alpha);
            else
                mu = gpr(gpdata.logtheta, gpdata.covfunc, gpdata.x, gpdata.y, X, gpdata.L, gpdata.alpha);
            end
        else
            if nargout == 2
                % Compute variance as well as mean
                [mu,varargout{2}] = gpr(gpdata.logtheta, gpdata.covfunc, gpdata.x, gpdata.y, X);
            else
                mu = gpr(gpdata.logtheta, gpdata.covfunc, gpdata.x, gpdata.y, X);
            end
        end
        
        % We consider that the mean is our best estimate. 
        varargout{1} = mu + gpdata.offset;
    case 'ComputeLForCaching'
        gpdata = varargin{1};
        K = feval(gpdata.covfunc{:}, gpdata.logtheta, X);   % compute training set covariance matrix
        try
            L = chol(K)';                            % cholesky factorization of the covariance
        catch
            K = IncreaseNoiseLevel(gpdata.covfunc, gpdata.logtheta, X);
            L = chol(K)';
        end
        varargout{1} = L;
        
    case 'EvaluateWithCachedL'
        % This method uses a cached factorised covariance matrix (L)
        gpdata = varargin{1};
        gpdata.alpha = solve_chol(gpdata.L',gpdata.y);
        varargout{1} = gaussianprocessregression('Evaluate',X,gpdata);
end







function [out1, out2] = gpr(logtheta, covfunc, x, y, xstar, varargin);

% gpr - Gaussian process regression, with a named covariance function. Two
% modes are possible: training and prediction: if no test data are given, the
% function returns minus the log likelihood and its partial derivatives with
% respect to the hyperparameters; this mode is used to fit the hyperparameters.
% If test data are given, then (marginal) Gaussian predictions are computed,
% whose mean and variance are returned. Note that in cases where the covariance
% function has noise contributions, the variance returned in S2 is for noisy
% test targets; if you want the variance of the noise-free latent function, you
% must substract the noise variance.
%
% usage: [nlml dnlml] = gpr(logtheta, covfunc, x, y)
%    or: [mu S2]  = gpr(logtheta, covfunc, x, y, xstar)
%    or: [mu S2]  = gpr(logtheta, covfunc, x, y, xstar, L, alpha)  % RF, 16/09/2009
%    
%
% where:
%
%   logtheta is a (column) vector of log hyperparameters
%   covfunc  is the covariance function
%   x        is a n by D matrix of training inputs
%   y        is a (column) vector (of size n) of targets
%   xstar    is a nn by D matrix of test inputs
%   nlml     is the returned value of the negative log marginal likelihood
%   dnlml    is a (column) vector of partial derivatives of the negative
%                 log marginal likelihood wrt each log hyperparameter
%   mu       is a (column) vector (of size nn) of prediced means
%   S2       is a (column) vector (of size nn) of predicted variances
%
% For more help on covariance functions, see "help covFunctions".
%
% (C) copyright 2006 by Carl Edward Rasmussen (2006-03-20).

if ischar(covfunc), covfunc = cellstr(covfunc); end % convert to cell if needed
[n, D] = size(x);
if eval(feval(covfunc{:})) ~= size(logtheta, 1) 
  errorMessage = ['Swordfish: Number of hyper-parameters in logtheta0 does not agree with covariance function (' num2str(eval(feval(covfunc{:}))) 'elements are necessary in logtheta0).'];
  if exist('midaslog', 'file'); midaslog('Warn', errorMessage); end;
  progress('Close', progressId)
  error(errorMessage);
end

if nargin == 7 % use precomputed and factorised training covariance matrix
    L = varargin{1};
    alpha = varargin{2};
else
    K = feval(covfunc{:}, logtheta, x);    % compute training set covariance matrix
    try
        L = chol(K)';                          % cholesky factorization of the covariance
    catch
        % K is probably singular 
        K = IncreaseNoiseLevel(covfunc, logtheta, x);
        L = chol(K)';
    end
    alpha = solve_chol(L',y);
end

if nargin == 4 % if no test cases, compute the negative log marginal likelihood

  out1 = 0.5*y'*alpha + sum(log(diag(L))) + 0.5*n*log(2*pi);

  if nargout == 2               % ... and if requested, its partial derivatives
    out2 = zeros(size(logtheta));       % set the size of the derivative vector
    W = L'\(L\eye(n))-alpha*alpha';                % precompute for convenience
    for i = 1:length(out2)
      out2(i) = sum(sum(W.*feval(covfunc{:}, logtheta, x, i)))/2;
    end
  end

else                    % ... otherwise compute (marginal) test predictions ...

  [Kss, Kstar] = feval(covfunc{:}, logtheta, x, xstar);     %  test covariances

  out1 = Kstar' * alpha;                                      % predicted means

  if nargout == 2
    v = L\Kstar;
    out2 = Kss - sum(v.*v)';
  end  

end



function [X, fX, i] = minimize(X, f, length, varargin)

% Minimize a differentiable multivariate function. 
%
% Usage: [X, fX, i] = minimize(X, f, length, P1, P2, P3, ... )
%
% where the starting point is given by "X" (D by 1), and the function named in
% the string "f", must return a function value and a vector of partial
% derivatives of f wrt X, the "length" gives the length of the run: if it is
% positive, it gives the maximum number of line searches, if negative its
% absolute gives the maximum allowed number of function evaluations. You can
% (optionally) give "length" a second component, which will indicate the
% reduction in function value to be expected in the first line-search (defaults
% to 1.0). The parameters P1, P2, P3, ... are passed on to the function f.
%
% The function returns when either its length is up, or if no further progress
% can be made (ie, we are at a (local) minimum, or so close that due to
% numerical problems, we cannot get any closer). NOTE: If the function
% terminates within a few iterations, it could be an indication that the
% function values and derivatives are not consistent (ie, there may be a bug in
% the implementation of your "f" function). The function returns the found
% solution "X", a vector of function values "fX" indicating the progress made
% and "i" the number of iterations (line searches or function evaluations,
% depending on the sign of "length") used.
%
% The Polack-Ribiere flavour of conjugate gradients is used to compute search
% directions, and a line search using quadratic and cubic polynomial
% approximations and the Wolfe-Powell stopping criteria is used together with
% the slope ratio method for guessing initial step sizes. Additionally a bunch
% of checks are made to make sure that exploration is taking place and that
% extrapolation will not be unboundedly large.
%
% See also: checkgrad 
%
% Copyright (C) 2001 - 2006 by Carl Edward Rasmussen (2006-09-08).

INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0;                  % extrapolate maximum 3 times the current step-size
MAX = 20;                         % max 20 function evaluations per line search
RATIO = 10;                                       % maximum allowed slope ratio
SIG = 0.1; RHO = SIG/2; % SIG and RHO are the constants controlling the Wolfe-
% Powell conditions. SIG is the maximum allowed absolute ratio between
% previous and new slopes (derivatives in the search direction), thus setting
% SIG to low (positive) values forces higher precision in the line-searches.
% RHO is the minimum allowed fraction of the expected (from the slope at the
% initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
% Tuning of SIG (depending on the nature of the function to be optimized) may
% speed up the minimization; it is probably not worth playing much with RHO.

% The code falls naturally into 3 parts, after the initial line search is
% started in the direction of steepest descent. 1) we first enter a while loop
% which uses point 1 (p1) and (p2) to compute an extrapolation (p3), until we
% have extrapolated far enough (Wolfe-Powell conditions). 2) if necessary, we
% enter the second loop which takes p2, p3 and p4 chooses the subinterval
% containing a (local) minimum, and interpolates it, unil an acceptable point
% is found (Wolfe-Powell conditions). Note, that points are always maintained
% in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction using
% conjugate gradients (Polack-Ribiere flavour), or revert to steepest if there
% was a problem in the previous line-search. Return the best value so far, if
% two consecutive line-searches fail, or whenever we run out of function
% evaluations or line-searches. During extrapolation, the "f" function may fail
% either with an error or returning Nan or Inf, and minimize should handle this
% gracefully.

if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
if length>0, S='Linesearch'; else S='Function evaluation'; end 

i = 0;                                            % zero the run length counter
ls_failed = 0;                             % no previous line search has failed
[f0 df0] = feval(f, X, varargin{:});          % get function value and gradient
fX = f0;
i = i + (length<0);                                            % count epochs?!
s = -df0; d0 = -s'*s;           % initial search direction (steepest) and slope
x3 = red/(1-d0);                                  % initial step is red/(|s|+1)

while i < abs(length)                                      % while not finished
  i = i + (length>0);                                      % count iterations?!

  X0 = X; F0 = f0; dF0 = df0;                   % make a copy of current values
  if length>0, M = MAX; else M = min(MAX, -length-i); end

  while 1                             % keep extrapolating as long as necessary
    x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0;
    success = 0;
    while ~success && M > 0
      try
        M = M - 1; i = i + (length<0);                         % count epochs?!
        [f3 df3] = feval(f, X+x3*s, varargin{:});
        if isnan(f3) || isinf(f3) || any(isnan(df3)+isinf(df3)), error(''), end
        success = 1;
      catch                                % catch any error which occured in f
        x3 = (x2+x3)/2;                                  % bisect and try again
      end
    end
    if f3 < F0, X0 = X+x3*s; F0 = f3; dF0 = df3; end         % keep best values
    d3 = df3'*s;                                                    % new slope
    if d3 > SIG*d0 || f3 > f0+x3*RHO*d0 || M == 0  % are we done extrapolating?
      break
    end
    x1 = x2; f1 = f2; d1 = d2;                        % move point 2 to point 1
    x2 = x3; f2 = f3; d2 = d3;                        % move point 3 to point 2
    A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);                 % make cubic extrapolation
    B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);
    x3 = x1-d1*(x2-x1)^2/(B+sqrt(B*B-A*d1*(x2-x1))); % num. error possible, ok!
    if ~isreal(x3) || isnan(x3) || isinf(x3) || x3 < 0 % num prob | wrong sign?
      x3 = x2*EXT;                                 % extrapolate maximum amount
    elseif x3 > x2*EXT                  % new point beyond extrapolation limit?
      x3 = x2*EXT;                                 % extrapolate maximum amount
    elseif x3 < x2+INT*(x2-x1)         % new point too close to previous point?
      x3 = x2+INT*(x2-x1);
    end
  end                                                       % end extrapolation

  while (abs(d3) > -SIG*d0 || f3 > f0+x3*RHO*d0) && M > 0  % keep interpolating
    if d3 > 0 || f3 > f0+x3*RHO*d0                         % choose subinterval
      x4 = x3; f4 = f3; d4 = d3;                      % move point 3 to point 4
    else
      x2 = x3; f2 = f3; d2 = d3;                      % move point 3 to point 2
    end
    if f4 > f0           
      x3 = x2-(0.5*d2*(x4-x2)^2)/(f4-f2-d2*(x4-x2));  % quadratic interpolation
    else
      A = 6*(f2-f4)/(x4-x2)+3*(d4+d2);                    % cubic interpolation
      B = 3*(f4-f2)-(2*d2+d4)*(x4-x2);
      x3 = x2+(sqrt(B*B-A*d2*(x4-x2)^2)-B)/A;        % num. error possible, ok!
    end
    if isnan(x3) || isinf(x3)
      x3 = (x2+x4)/2;               % if we had a numerical problem then bisect
    end
    x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2));  % don't accept too close
    [f3 df3] = feval(f, X+x3*s, varargin{:});
    if f3 < F0, X0 = X+x3*s; F0 = f3; dF0 = df3; end         % keep best values
    M = M - 1; i = i + (length<0);                             % count epochs?!
    d3 = df3'*s;                                                    % new slope
  end                                                       % end interpolation

  if abs(d3) < -SIG*d0 && f3 < f0+x3*RHO*d0          % if line search succeeded
    X = X+x3*s; f0 = f3; fX = [fX' f0]';                     % update variables
    fprintf('%s %6i;  Value %4.6e\r', S, i, f0);
    s = (df3'*df3-df0'*df3)/(df0'*df0)*s - df3;   % Polack-Ribiere CG direction
    df0 = df3;                                               % swap derivatives
    d3 = d0; d0 = df0'*s;
    if d0 > 0                                      % new slope must be negative
      s = -df0; d0 = -s'*s;                  % otherwise use steepest direction
    end
    x3 = x3 * min(RATIO, d3/(d0-realmin));          % slope ratio but max RATIO
    ls_failed = 0;                              % this line search did not fail
  else
    X = X0; f0 = F0; df0 = dF0;                     % restore best point so far
    if ls_failed || i > abs(length)         % line search failed twice in a row
      break;                             % or we ran out of time, so we give up
    end
    s = -df0; d0 = -s'*s;                                        % try steepest
    x3 = 1/(1-d0);                     
    ls_failed = 1;                                    % this line search failed
  end
end
fprintf('\n');

function [A, B] = covSum(covfunc, logtheta, x, z);

% covSum - compose a covariance function as the sum of other covariance
% functions. This function doesn't actually compute very much on its own, it
% merely does some bookkeeping, and calls other covariance functions to do the
% actual work.
%
% For more help on design of covariance functions, try "help covFunctions".
%
% (C) Copyright 2006 by Carl Edward Rasmussen, 2006-03-20.

for i = 1:length(covfunc)                   % iterate over covariance functions
  f = covfunc(i);
  if iscell(f{:}), f = f{:}; end          % dereference cell array if necessary
  j(i) = cellstr(feval(f{:}));
end

if nargin == 1,                                   % report number of parameters
  A = char(j(1)); for i=2:length(covfunc), A = [A, '+', char(j(i))]; end
  return
end

[n, D] = size(x);

v = [];              % v vector indicates to which covariance parameters belong
for i = 1:length(covfunc), v = [v repmat(i, 1, eval(char(j(i))))]; end

switch nargin
case 3                                              % compute covariance matrix
  A = zeros(n, n);                       % allocate space for covariance matrix
  for i = 1:length(covfunc)                  % iteration over summand functions
    f = covfunc(i);
    if iscell(f{:}), f = f{:}; end        % dereference cell array if necessary
    A = A + feval(f{:}, logtheta(v==i), x);            % accumulate covariances
  end

case 4                      % compute derivative matrix or test set covariances
  if nargout == 2                                % compute test set cavariances
    A = zeros(size(z,1),1); B = zeros(size(x,1),size(z,1));    % allocate space
    for i = 1:length(covfunc)
      f = covfunc(i);
      if iscell(f{:}), f = f{:}; end      % dereference cell array if necessary
      [AA BB] = feval(f{:}, logtheta(v==i), x, z);   % compute test covariances
      A = A + AA; B = B + BB;                                  % and accumulate
    end
  else                                            % compute derivative matrices
    i = v(z);                                       % which covariance function
    j = sum(v(1:z)==i);                    % which parameter in that covariance
    f = covfunc(i);
    if iscell(f{:}), f = f{:}; end        % dereference cell array if necessary
    A = feval(f{:}, logtheta(v==i), x, j);                 % compute derivative
  end

end


function [A, B] = covSEard(loghyper, x, z)

% Squared Exponential covariance function with Automatic Relevance Detemination
% (ARD) distance measure. The covariance function is parameterized as:
%
% k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
%
% where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
% D is the dimension of the input space and sf2 is the signal variance. The
% hyperparameters are:
%
% loghyper = [ log(ell_1)
%              log(ell_2)
%               .
%              log(ell_D)
%              log(sqrt(sf2)) ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% (C) Copyright 2006 by Carl Edward Rasmussen (2006-03-24)

if nargin == 0, A = '(D+1)'; return; end          % report number of parameters

persistent K;    

[n D] = size(x);
ell = exp(loghyper(1:D));                         % characteristic length scale
sf2 = exp(2*loghyper(D+1));                                   % signal variance

if nargin == 2
  
    % Original code, commented out by CL 10/9/09
    % K = sf2*exp(-sq_dist(diag(1./ell)*x')/2);
   
    % Replacement call to MEX function. This seems to be about 7 times
    % faster by exploiting the symmetry of K and writing it all in C
    K = computesek(ell, x, sf2);
  
  A = K;                 
elseif nargout == 2                              % compute test set covariances
  A = sf2*ones(size(z,1),1);

  % Original code, commented out by CL 16/9/09 
  % B = sf2*exp(-sq_dist(diag(1./ell)*x',diag(1./ell)*z')/2);

  % Replacement call to MEX function. This seems to be about 2 times
  % faster: the improvement is less dramatic than above because we can't
  % exploit symmetry.
  B = computesek(ell, x, z, sf2);
  
else                                                % compute derivative matrix
  
  % check for correct dimension of the previously calculated kernel matrix
  if any(size(K)~=n)  
    % Same deal again: original code commented out by CL, 16/9/09, and
    % replaced by call to COMPUTESEK.
    % K = sf2*exp(-sq_dist(diag(1./ell)*x')/2);
    K = computesek(ell, x, sf2);
  end
   
  if z <= D                                           % length scale parameters
    A = K.*sq_dist(x(:,z)'/ell(z));  
  else                                                    % magnitude parameter
    A = 2*K;
    clear K;
  end
end


function [A, B] = covNoise(logtheta, x, z);

% Independent covariance function, ie "white noise", with specified variance.
% The covariance function is specified as:
%
% k(x^p,x^q) = s2 * \delta(p,q)
%
% where s2 is the noise variance and \delta(p,q) is a Kronecker delta function
% which is 1 iff p=q and zero otherwise. The hyperparameter is
%
% logtheta = [ log(sqrt(s2)) ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% (C) Copyright 2006 by Carl Edward Rasmussen, 2006-03-24.

if nargin == 0, A = '1'; return; end              % report number of parameters

s2 = exp(2*logtheta);                                          % noise variance

if nargin == 2                                      % compute covariance matrix
  A = s2*eye(size(x,1));
elseif nargout == 2                              % compute test set covariances
  A = s2;
  B = 0;                               % zeros cross covariance by independence
else                                                % compute derivative matrix
  A = 2*s2*eye(size(x,1));
end


function C = sq_dist(a, b, Q);

if nargin < 1 | nargin > 3 | nargout > 1
  error('Wrong number of arguments.');
end

if nargin == 1 | isempty(b)                   % input arguments are taken to be
  b = a;                                   % identical if b is missing or empty
end 

[D, n] = size(a); 
[d, m] = size(b);
if d ~= D
  error('Error: column lengths must agree.');
end

if nargin < 3
  C = zeros(n,m);
  for d = 1:D
    C = C + (repmat(b(d,:), n, 1) - repmat(a(d,:)', 1, m)).^2;
  end
  % C = repmat(sum(a.*a)',1,m)+repmat(sum(b.*b),n,1)-2*a'*b could be used to 
  % replace the 3 lines above; it would be faster, but numerically less stable.
else
  if [n m] == size(Q)
    C = zeros(D,1);
    for d = 1:D
      C(d) = sum(sum((repmat(b(d,:), n, 1) - repmat(a(d,:)', 1, m)).^2.*Q));
    end
  else
    error('Third argument has wrong size.');
  end
end


function x = solve_chol(A, B);

if nargin ~= 2 | nargout > 1
  error('Wrong number of arguments.');
end

if size(A,1) ~= size(A,2) | size(A,1) ~= size(B,1)
  error('Wrong sizes of matrix arguments.');
end

x = A\(A'\B);


function [f, g, dfdx, dgdx] = knitrowrap(x, covfunc, X, y)
%OBJFCN - Function in the right format for use as an objective 
%   function for mknitro. 

[g,dgdx] = gpr(x, covfunc, X, y);

% Sometimes computing the derivatives costs more than letting knitro
% perform finite differences
% g= gpr(x, covfunc, X, y);
% dgdx = [];

% Unconstrained
f = [];
dfdx = [];


function K = IncreaseNoiseLevel(covfunc,logtheta,X)
% In some cases where noise appears to be very low, the covariance matrix K
% does not have full rank. This casuses the Cholesky decomposition to fail.
% Here we artificially saturate the noise level at a minimum bound.
% RF, 12/03/2010

MINLOGNOISE = -6;

if logtheta(end)<MINLOGNOISE; logtheta(end)=MINLOGNOISE; end
K = feval(covfunc{:}, logtheta, X);


function logtheta = EstimateHyperparameters(logtheta0, gpoptions, X, y)
% Estimates the hyperparameters  by maximising the log marginal likelihood
% from a starting point logtheta0.
%
% RF, 31/05/2010

dimInputs = size(X,2);

tic
% Use gpml minimize function
if isfield(gpoptions,'numMaxFunEvaluations')
    numMaxFunEvaluations = gpoptions.numMaxFunEvaluations;
else
    numMaxFunEvaluations = 50;
end
[logtheta, fvals, iter] = minimize(logtheta0, 'gpr', -numMaxFunEvaluations, ...
    gpoptions.covfunc, X, y);
disp(['Hyperparameters computed in ' num2str(toc) ' seconds.'])
disp('Logarithm of the estimated value of the hyperparameters:')
disp(logtheta)



