function [x u]=estimate_x_and_u(F, N, S, m, E, d, v, u, z, y, x, spk_ids)

% ESTIMATE_X_AND_U estimates channel factors and eigenchannels for
% joint factor analysis model 
%
%
% [x u]=estimate_x_and_u(F, N, S, m, E, d, v, u, z, y, x, spk_ids)
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
% z - matrix of speaker factors corresponding to matrix d. The rows
%     correspond to speakers (values in vector spk_ids are the indices of the
%     rows, therfore the number of the rows must be (at least) the highest
%     value in spk_ids). Number of columns is given by the supervector 
%     dimensionality.
% x - NOT USED by this function; used by other JFA function as
%     matrix of channel factors. The rows correspond to training
%     segments. The columns correspond to eigenchannels (The number of columns
%     must be the same as the number of rows of matrix u)
% spk_ids - column vector with rows corresponding to training segments and
%     integer values identifying a speaker. Rows having same values identifies
%     segments spoken by same speakers. The values are indices of rows in
%     y and z matrices containing corresponding speaker factors.
%
%
% x=estimate_x_and_u(F, N, S, m, E, d, v, u, z, y, x, spk_ids)
%
% only the channels factors are estimated
%
%
% [x A C]=estimate_x_and_u(F, N, S, m, E, d, v, u, z, y, x, spk_ids)
%
% estimates channels factors and acumulators A and C. A is cell array of MxM
% matrices, where M is number of eigenchannels. Number of elements in the
% cell array is given by number of Gaussian components. C is of the same size
% at the matrix u.
%
%
% u=estimate_x_and_u(A, C)
%
% updates eigenchannels from accumulators A and C. Using F and N statistics
% corresponding to subsets of training segments, multiple sets of accumulators
% can be collected (possibly in parallel) and summed before the update. Note
% that segments of one speaker must not be split into different subsets.

if nargin == 2 && nargout == 1
  % update u from acumulators A and C
  x=update_u(F, N);
  return
end

% this will just create a index map, so that we can copy the counts n-times (n=dimensionality)
dim = size(F,2)/size(N,2);
index_map = reshape(repmat(1:size(N,2), dim,1),size(F,2),1);
x = zeros(size(spk_ids,1), size(u,1));

if nargout > 1
  for c=1:size(N,2)
    A{c} = zeros(size(u,1));
  end  
  C = zeros(size(u,1), size(F,2));
end

for c=1:size(N,2)
  c_elements = ((c-1)*dim+1):(c*dim);
  uEuT{c} = u(:,c_elements) .* repmat(1./E(c_elements), size(u,1), 1) * u(:,c_elements)';
end

for ii = unique(spk_ids)'
  speakers_sessions = find(spk_ids == ii);
  spk_shift = m + y(ii,:) * v + z(ii,:) .* d;
  for jj = speakers_sessions'
    Nh = N(jj, index_map);
    Fh = F(jj,:) - Nh .* spk_shift;

%   L = eye(size(u,1)) + u * diag(Nh./E) * u';
    L = eye(size(u,1));
    for c=1:size(N,2)
      L = L + uEuT{c} * N(jj,c);
    end

    invL = inv(L);
    x(jj,:) = ((Fh./E) * u') * invL;

    if nargout > 1
      invL = invL + x(jj,:)' * x(jj,:);
      for c=1:size(N,2)
        A{c} = A{c} + invL * N(jj,c);
      end
      C = C + x(jj,:)' * Fh;
    end
  end
end

if nargout == 3
 % output new estimates of x and accumulators A and C
 u = A;
elseif nargout == 2
 % output new estimates of x and u
 u=update_u(A, C);
end

%-------------------------------------------------
function u=update_u(A, C)
u = zeros(size(C));
dim = size(C,2)/length(A);
for c=1:length(A)
  c_elements = ((c-1)*dim+1):(c*dim);
  u(:,c_elements) = inv(A{c}) * C(:,c_elements);
end
