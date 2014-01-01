function [scores] = kscore_famous_19_mod1(F, N, S, m, E, d, v, u, z, y, x, scores)

% kscore_famous_19 produces matrix of scores for the given models and
% utterances (in the form of 0th and 1st order statistic)
%
%
% [x u]=kscore_famous_19(F, N, S, m, E, d, v, u, z, y, x, scores)
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
% x - NOT USED by this function; used by other JFA function as
%     matrix of channel factors. The rows correspond to training
%     segments. The columns correspond to eigenchannels (The number of columns
%     must be the same as the number of rows of matrix u)
% scores - scores mask matrix; on each "1" position, a score will be produced
%     if scores is sparse, the output will also be sparse

dim  = size(F,1)/size(N,1);
index_map = reshape(repmat(1:size(N,1), dim,1),size(F,1),1);

% synthesize models
M = repmat(m, 1, size(y,2))  + z .* repmat(d, 1, size(y, 2)) + v*y;

% add the UBM to the begining
M = [m M];

% precompute u'*E^-1*u
for c=1:size(N,1)
  c_elements = ((c-1)*dim+1):(c*dim);
  uEuT{c} = u(c_elements,:)' * (repmat(1./E(c_elements), 1, size(u,2)) .* u(c_elements,:));
end

sum_N = sum(N, 1);
scores = [ ones(1, size(scores,2)); scores ];
whos sum_N scores

% go over all testing utterances
for ii = 1:size(F,2)
  Nt  = N(index_map,ii);
  Nte = Nt ./ E;
  Fte = F(:,ii) ./ E;

  % compute l
  L = eye(size(u,2));
  for c=1:size(N,1)
    L = L + uEuT{c} * N(c,ii);
  end

  cholLu  = chol(L,'lower') \ u';

  for jj = 1:size(M,2)
    if scores(jj, ii) == 1
      MNe    = Nte .* M(:,jj);
      Fse    = Fte - MNe;

      lin    = Fte' * M(:,jj);
      quad   = MNe' * M(:,jj);
      quad2  = cholLu * Fse; 
      quad2  = quad2' * quad2; 

      scores(jj,ii) = (lin - 0.5*quad + 0.5*quad2) / sum_N(ii);
      % scores(jj,ii) = (lin - 0.5*quad + 0.5*quad2);
    end
  end
end

% substract the UBM score
ubm_score = scores(1,:)';
for ii=1:size(scores,2)
  for jj=1:size(scores,1)
    if scores(jj,ii) ~= 0
      scores(jj,ii) = scores(jj,ii) - ubm_score(ii,1);
    end
  end
end

scores(1,:) = [];

