function [N F] = collect_suf_stats(data, m, v, w)
%Collect sufficient stats in baum-welch fashion
%  [N, F] = collect_suf_stats(FRAMES, M, V, W) returns the vectors N and
%  F of zero- and first- order statistics, respectively, where FRAMES is a
%  dim x length matrix of features, M is dim x gaussians matrix of GMM means
%  V is a dim x gaussians matrix of GMM variances, W is a vector of GMM weights.

n_mixtures  = size(w, 1);
dim         = size(m, 1);

% compute the GMM posteriors for the given data
gammas = gaussian_posteriors(data, m, v, w);
% gammas_size = size(gammas)   512 x 12331

% zero order stats for each Gaussian are just sum of the posteriors (soft counts)
N = sum(gammas,2);
%size_N = size(N)		% 512 x 1

% first order stats is just a (posterior) weighted sum
F = data * gammas';
size_F_orig = size(F);	% 39 x 512
F = reshape(F, n_mixtures*dim, 1);
size_F = size(F);		% 19968 x 1

