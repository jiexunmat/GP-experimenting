function [y] = jointSamplingFromGP(x,hyp)
    n_samples = size(x,1);
    n_dim = size(x,2);
    
    K = covSEbasic(x,x,hyp);

    z = randn(n_samples, n_dim);
    m = zeros(n_samples, n_dim);
    %y = chol(K)'*z;
    y = chol(K + 1e-6*eye(n_samples))'*z + m;
end