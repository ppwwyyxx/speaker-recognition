function gammas = gaussian_posteriors(data, m, v, w)
% Computes Gaussian posterior probs for the given data
%   For each frame (column) of data, compute the vector of posterior probs
%   of the given GMM given by means, vars, and weights

n_mixtures  = size(w, 1);				% 512
n_frames    = size(data, 2);			% 12231 frames
dim         = size(data, 1);			% 39


% precompute the model g-consts and normalize the weights
a = w ./ (((2*pi)^(dim/2)) * sqrt(prod(v)'));

gammas = zeros(n_mixtures, n_frames);

for ii =1:n_mixtures
  gammas(ii,:) = gaussian_function(data, a(ii), m(:,ii), v(:,ii));
end
gamasum = sum(gammas);		% 1 x 12331

% normalize
gammas=bsxfun(@rdivide, gammas, gamasum);
%gammas
