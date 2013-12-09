%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
niter = 300

% initial eigenvoices and eigenchannels matrices
eigen_channels_mat_file = 'output/u_final.mat';
eigen_voices_mat_file   = 'output/v_final.mat';

% training data mat file
% (F, N, spk_logical, spk_ids)
train_mat_file = './output/train.mat'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RUN
% load the training data matrix
% N ... zero order sufficient stats
% F ... first order sufficient stats
% spk_ids ... numerical speaker identification (easy for matlab to do uniqe on
%             these)
disp(['Loading training stats from ' train_mat_file])
load(train_mat_file); %loads F, N, spk_ids

% load initial u and v matrices
disp(['Loading initial u matrix from ' eigen_channels_mat_file])
load(eigen_channels_mat_file);

disp(['Loading initial v matrix from ' eigen_voices_mat_file])
load(eigen_voices_mat_file);

disp(['Initializing d']);
d = randn(1, size(F, 2)) * sum(E,2) * 0.001;

% we don't use the second order stats - set 'em to empty matrix
S = [];

% we need the ubm params
% m ... supervector of means
% E ... supervector of variances
m = load(['models/ubm_means'], '-ascii');
E = load(['models/ubm_variances'], '-ascii');

[junk junk spk_ids]=unique(spk_ids);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_speakers=max(spk_ids);
n_sessions=size(spk_ids,1);


% estimate speaker and channel factors. these will be fixed througout the d
% estimation. note that for estimating y, we don't use any channel info; this
% is analogous to v estimation, where no channel info was used.
[y]=estimate_y_and_v(F, N, S, m, E, 0, v, 0, zeros(n_speakers,1), 0, zeros(n_sessions,1), spk_ids);
[x]=estimate_x_and_u(F, N, S, m, E, 0, v, u, zeros(n_speakers,1), y, zeros(n_sessions,1), spk_ids);

% estimate speaker factors. these will be fixed througout the u estimation
% [y]=estimate_y_and_v(F, N, S, m, E, 0, v, 0, zeros(n_speakers,1), 0, zeros(n_sessions,1), spk_ids);

% iteratively retrain u
for ii=1:niter
  disp(' ')
  disp(['Starting iteration: ' num2str(ii)])

  ct=cputime;
  et=clock;

  % go baby...
  % note, that we don't use the reestimated y vectors (we want to keep the spk
  % direction constant and concentrate on the channel subspace)
  [z d]=estimate_z_and_d(F, N, S, m, E, d, v, u, zeros(n_speakers,1), y, x, spk_ids);

  disp(['Iteration: ' num2str(ii) ' CPU time:' num2str(cputime-ct) ' Elapsed time: ' num2str(etime(clock,et))])

  if ii == niter
	  out_final = ['output/d_final']
	  save(out_final, 'd')
  end
end

