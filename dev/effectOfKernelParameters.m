clear all;
% Basic parameters
n_samples = 50;

% Pick some points in x
lb = -5; ub = 5;
x = linspace(-5,5,n_samples);
x = x';

% Initialise several length scales
ell = [0.1, 1, 5]; 
sf = 1;

figure;
title('Effect of varying the length scale');
hold on; 
for i = 1:length(ell)
    hyp = [ell(i); sf];
    y = jointSamplingFromGP(x,hyp);
    plot(x,y);
    legendInfo{i} = ['ell = ' num2str(ell(i))];
end
legend(legendInfo);

% Initialise several signal noise
ell = 1; 
sf = [0.1, 1, 5];

figure;
title('Effect of varying signal noise');
hold on; 
for i = 1:length(sf)
    hyp = [ell; sf(i)];
    y = jointSamplingFromGP(x,hyp);
    plot(x,y);
    legendInfo{i} = ['sf = ' num2str(sf(i))];
end
legend(legendInfo);