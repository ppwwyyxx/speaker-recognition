function [y v C]=estimate_y_and_v(F, N, S, m, E, d, v, u, z, y, x, spk_ids)

% ESTIMATE_Y_AND_V estimates speaker factors and eigenvoices for
% joint factor analysis model 
%
%
% [y v]=estimate_y_and_v(F, N, S, m, E, d, v, u, z, y, x, spk_ids)
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
% y - NOT USED by this function; used by other JFA function as
%     matrix of speaker factors corresponding to eigenvoices. The rows
%     correspond to speakers (values in vector spk_ids are the indices of the
%     rows, therfore the number of the rows must be (at least) the highest
%     value in spk_ids). The columns correspond to eigenvoices (The number
%     of columns must the same as the number of rows of matrix v).
% z - matrix of speaker factors corresponding to matrix d. The rows
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
% y=estimate_y_and_v(F, N, S, m, E, d, v, u, z, y, x, spk_ids)
%
% only the speaker factors are estimated
%
%
% [y A C]=estimate_y_and_v(F, N, S, m, E, d, v, u, z, y, x, spk_ids)
%
% estimates speaker factors and acumulators A and C. A is cell array of MxM
% matrices, where M is number of eigenvoices. Number of elements in the
% cell array is given by number of Gaussian components. C is of the same size
% at the matrix v.
%
%
% v=estimate_y_and_v(A, C)
%
% updates eigenvoices from accumulators A and C. Using F and N statistics
% corresponding to subsets of training segments, multiple sets of accumulators
% can be collected (possibly in parallel) and summed before the update. Note
% that segments of one speaker must not be split into different subsets.

if nargin == 2 && nargout == 1
  % update v from acumulators A and C
  y=update_v(F, N);
  return
end

% this will just create a index map, so that we can copy the counts n-times (n=dimensionality)
dim = size(F,2)/size(N,2);
index_map = reshape(repmat(1:size(N,2), dim,1),size(F,2),1);
y = zeros(max(spk_ids), size(v,1));

if nargout > 1
  for c=1:size(N,2)
    A{c} = zeros(size(v,1));
  end
  C = zeros(size(v,1), size(F,2));
end

for c=1:size(N,2)
  c_elements = ((c-1)*dim+1):(c*dim);
  vEvT{c} = v(:,c_elements) .* repmat(1./E(c_elements), size(v, 1), 1) * v(:,c_elements)';
end
for ii = unique(spk_ids)'
  speakers_sessions = find(spk_ids == ii);
  Fs = sum(F(speakers_sessions,:), 1);
  Nss = sum(N(speakers_sessions,:), 1);
  Ns = Nss(1,index_map);
  Fs = Fs -  (m + z(ii,:) .* d) .* Ns;
  for jj = speakers_sessions'
    Fs = Fs - (x(jj,:) * u) .* N(jj, index_map);
  end

% L = eye(size(v,1)) + v * diag(Ns./E) * v';
  L = eye(size(v,1));
  for c=1:size(N,2)
    L = L + vEvT{c} * Nss(c);
  end

  invL = inv(L);
  y(ii,:) = ((Fs ./ E) * v') * invL;
  if nargout > 1
    invL = invL + y(ii,:)' * y(ii,:);
    for c=1:size(N,2)
      A{c} = A{c} + invL * Nss(c);
    end
    C = C + y(ii,:)' * Fs;
  end
end  

if nargout == 3
 % output new estimates of y and accumulators A and C
 v = A;
elseif nargout == 2
 % output new estimates of y and v
 v=update_v(A, C);
end

%-------------------------------------------------
function C=update_v(A, C)
dim = size(C,2)/length(A);
for c=1:length(A)
  c_elements = ((c-1)*dim+1):(c*dim);
  C(:,c_elements) = inv(A{c}) * C(:,c_elements);
end
