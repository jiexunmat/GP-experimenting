function [ K ] = covSEbasic( X,Z,hyp )
%COVSEbasic uses a simple squared exponential covariance function
% X.shape = (m,dim)
% Z.shape = (n,dim)
% K.shape = (m,n)

dim_x = size(X,2);
dim_z = size(Z,2);
assert(dim_x==dim_z, 'Dimensionality of x & z are not the same.');
assert(length(hyp)==dim_x + 1, 'The number of hyperparameters do not match.');

% Extract kernel hyperparameters
ell = hyp(1:end-1);
sf = hyp(end);

a = diag(1./ell) * X';
b = diag(1./ell) * Z';

K = sf.^2 .* exp(-sqrdist(a,b)/2);

end

function [ dist ] = sqrdist( a,b )
    [D,m] = size(a);
    [~,n] = size(b);
    dist = zeros(m,n);       % dist.shape = K.shape
    for i = 1:D              % accumulate sqr dist by dimensions
        a1 = a(i,:);
        b1 = b(i,:);
        dist = dist + (repmat(a1',1,n) - repmat(b1,m,1)).^2;
    end
end