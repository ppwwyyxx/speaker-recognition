function Y = gaussian_function(data, a, b, c)
%Y = gaus(data, b, c)
%GAUS N-domensional gaussian function---
%    See http://en.wikipedia.org/wiki/Gaussian_function for definition.
%    note it's not Gaussian distribution as no normalization (g-const) is performed
%
%    Every row data is one input vector. Y is column vector with the
%    same number of rows as data

N   = size(data,2);
dim = size(data,1);

auxC = -0.5 ./ c;

aux = bsxfun(@minus, data, b);
aux = aux .^ 2;
Y = auxC' *aux;

Y = exp(Y) * a;

