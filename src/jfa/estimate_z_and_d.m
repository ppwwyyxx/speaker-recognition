function [z d b]=estimate_z_and_d(F, N, S, m, E, d, v, u, z, y, x, spk_ids)

% ESTIMATE_Z_AND_D esimates vector that is diagonal of the diagonal matrix
%     describing the remaining speaker variability not described by 
%     eigenvoices. Also estimates matrix of corresponding speaker factors.
%
%
% [z d]=estimate_z_and_d(F, N, S, m, E, d, v, u, z, y, x, spk_ids)
%
% provides new estimates of channel factors, x, and 'eigenchannels', u,
% given zeroth and first order sufficient statistics (N. F), current
% hyper-parameters of joint factor analysis  model (m, E, d, u, v) and
% current estimates of speaker and channel factors (x, y, z)
%
% F - matrix of first order statistics (not centered). The rows correspond
%     to training segments. Number of columns is given by the supervector
%     dimensionality. The first n collums correspond to the n dimensions
%     of the first Gaussian component, the second n collums to second 
%     component, and so on.
% N - matrix of zero order statistics (occupation counts of Gaussian
%     components). The rows correspond to training segments. The collums
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
% z - NOT USED by this function; used by other JFA function as
%     matrix of speaker factors corresponding to matrix d. The rows
%     correspond to speakers (values in vector spk_ids are the indices of the
%     rows, therfore the number of the rows must be (at least) the highest
%     value in spk_ids). Number of columns is given by the supervector 
%     dimensionality.
% x - matrix of channel factors. The rows correspond to training
%     segments. The columns correspond to eigenchannels (The number of columns
%     must be the same as the number of rows of matrix u)
% spk_ids - column vector with rows corresponding to training segments and
%     integer values identifying a speaker. Rows having same values identifies
%     segments spoken by same speakers. The values are indices of rows in
%     y and z matrices containing corresponding speaker factors.
%
%
% z=estimate_z_and_d(F, N, S, m, E, d, v, u, z, y, x, spk_ids)
%
% only the speaker factors are estimated
%
%
% [z a b]=estimate_z_and_d(F, N, S, m, E, d, v, u, z, y, x, spk_ids)
%
% estimates speaker factors and acumulators a and b. Both a and b are vectors
% of the same size as vector d.
%
%
% d=estimate_z_and_d(a, b)
%
% updates d from accumulators a and b. Using F and N statistics
% corresponding to subsets of training segments, multiple sets of accumulators
% can be collected (possibly in parallel) and summed before the update. Note
% that segments of one speaker must not be split into different subsets.

if nargin == 2 && nargout == 1
  % update d from acumulators a and b
  z = N ./ F;
  return
end

% this will just create a index map, so that we can copy the counts n-times (n=dimensionality)
index_map = reshape(repmat(1:size(N,2),size(F,2)/size(N,2),1),size(F,2),1);
a = zeros(1, size(F,2));
b = zeros(1, size(F,2));
z = zeros(max(spk_ids), size(F,2));

for ii = unique(spk_ids)'
  speakers_sessions = find(spk_ids == ii);
  Fs = sum(F(speakers_sessions,:), 1);
  Ns = sum(N(speakers_sessions, index_map), 1);

  shift = m + y(ii,:) * v;
  Fs = Fs -  shift      .* Ns;
  for jj = speakers_sessions'
    shift = x(jj,:) * u;
    Fs = Fs - shift .* N(jj, index_map);
  end
  L = ones(1, size(d, 2)) + Ns ./ E .* (d.^2);
  z(ii,:) = Fs ./ E .* d ./L;
  
  if nargout > 1
    a = a + (1./L + z(ii,:).^2) .* Ns;
    b = b + z(ii,:) .* Fs;
  end    
end

if nargout == 3
 % output new estimates of z and accumulators a and b
 d = a;
elseif nargout == 2
 % output new estimates of z and d
 d = b ./ a;
end
