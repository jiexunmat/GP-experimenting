function S = lhs(xMin,xMax,nSample)
% LHS  Latin Hypercube Sampling
% Input:
%   xMin    : vector of minimum bounds
%   xMax    : vector of maximum bounds
%   nSample : number of samples
% Output:
%   S       : matrix containing the sample (nSample,numVar)
%
% RF, 20/04/2009

numVar = length(xMin);
ran = rand(nSample,numVar);
S = zeros(nSample,numVar);
for i=1:numVar
   idx = randperm(nSample);
   P = (idx'-ran(:,i))/nSample;
   S(:,i) = xMin(i) + P.* (xMax(i)-xMin(i));
end
S = S(randperm(size(S,1)),:);
