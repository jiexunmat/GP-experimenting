function y = UF1(x)
    [n_dim, num]  = size(x);
    tmp         = zeros(n_dim,num);
    tmp(2:n_dim,:)= (x(2:n_dim,:) - sin(6.0*pi*repmat(x(1,:),[n_dim-1,1]) + pi/n_dim*repmat((2:n_dim)',[1,num]))).^2;
    tmp1        = sum(tmp(3:2:n_dim,:));  % odd index
    tmp2        = sum(tmp(2:2:n_dim,:));  % even index
    y(1,:)      = x(1,:)             + 2.0*tmp1/size(3:2:n_dim,2);
    y(2,:)      = 1.0 - sqrt(x(1,:)) + 2.0*tmp2/size(2:2:n_dim,2);
    % disp(size(x));
    % disp(size(y));
    clear tmp;
end


