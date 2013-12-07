function [scores] = linear_scoring(F, N, S, m, E, d, v, u, z, y, x, scores)

% linear scoring produces matrix of scores for the given models and
% utterances (in the form of 0th and 1st order statistic)
%
%
% [scores]=linear_scoring(F, N, S, m, E, d, v, u, z, y, x)
%
% produces a MxN matrix of scores for utterances (given by N and F) and
%
% F - matrix of first order statistics (not centered). The cols correspond
%     to training segments. Number of rows is given by the supervector
%     dimensionality. The first n rows correspond to the n dimensions
%     of the first Gaussian component, the second n rows to second
%     component, and so on.
% N - matrix of zero order statistics (occupation counts of Gaussian
%     components). The rows correspond to training segments. The rows
%     correspond to Gaussian components.
% S - NOT USED by this function; reserved for second order statistics
% m - speaker and channel independent mean supervector (e.g. concatenated
%     UBM mean vectors)
% E - speaker and channel independent variance supervector (e.g. concatenated
%     UBM variance vectors)
% d - Row vector that is the diagonal from the diagonal matrix describing the
%     remaining speaker variability (not described by eigenvoices). Number of
%     columns is given by the supervector dimensionality.
% v - The rows of matrix v are 'eigenvoices'. (The number of rows must be the
%     same as the number of columns of matrix y). Number of columns is given
%     by the supervector dimensionality.
% u - The rows of matrix u are 'eigenchannels'. (The number of rows must be
%     the same as the number of columns of matrix x) Number of columns is
%     given by the supervector dimensionality.
% y - matrix of speaker factors corresponding to eigenvoices. The rows
%     correspond to speakers (values in vector spk_ids are the indices of the
%     rows, therfore the number of the rows must be (at least) the highest
%     value in spk_ids). The columns correspond to eigenvoices (The number
%     of columns must the same as the number of rows of matrix v).
% z - matrix of speaker factors corresponding to matrix d. The rows
%     correspond to speakers (values in vector spk_ids are the indices of the
%     rows, therfore the number of the rows must be (at least) the highest
%     value in spk_ids). Number of columns is given by the supervector
%     dimensionality.
% x - matrix of channel factors corresponding to eigenchannels
%     The rows correspond to training
%     segments. The columns correspond to eigenchannels (The number of columns
%     must be the same as the number of rows of matrix u)
%scores - score mask not used

dim  = size(F,2)/size(N,2);		% 39
%index_map = reshape(repmat(1:size(N,1), dim,1),size(F,1),1);

% models centered to UBM
M = z .* repmat(d, size(z, 1), 1) + y * v;

% models divided by variances
M = M ./ repmat(E, size(M, 1), 1);

% global channel shift computed on UBM
channel_shifts = repmat(m, size(F,1), 1) + x * u ;

%channel compensate 1st order stat
index_map = reshape(repmat(1:size(N,2),size(F,2)/size(N,2),1),size(F,2),1);
for ii=1:size(channel_shifts,2)
	channel_shifts(:,ii) = channel_shifts(:,ii) .* N(:, index_map(ii));
end
F = F - channel_shifts;

sum_N = repmat(sum(N, 2), 1, size(F, 2));
F = F ./ sum_N;

scores = M*F';
