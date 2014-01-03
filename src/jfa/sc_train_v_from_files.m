%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INIT THE EXPERIMENT
ny = 300;           % number of eigenvoices
niter = 300;

% initial eigenvoices matrix
% eigen_voices_mat_file   = ['output/v_pca' num2str(ny, '%03d') '.mat'];

% training data mat file
% (F, N, spk_logical, spk_ids)
train_mat_file = './output/train.mat'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RUN
% load the training matrix
% N ... zero order sufficient stats
% F ... first order sufficient stats
% spk_ids ... numerical speaker identification (easy for matlab to do uniqe on
%             these)
disp(['Loading training stats from ' train_mat_file])
load(train_mat_file); %loads F, N, spk_logical(spk_ids)

% we need the ubm params
% m ... supervector of means
% E ... supervector of variances
m = load(['models/ubm_means'], '-ascii');
E = load(['models/ubm_variances'], '-ascii');

[junk junk spk_ids]=unique(spk_logical);

% load initial v matrix
% disp(['Loading initial v matrix from ' eigen_voices_mat_file])
% load(eigen_voices_mat_file);

% or do not load initial matrix, but rather initialize it randomly
disp(['Initializing V matrix (randomly)'])
v = randn(ny, size(F, 2)) * sum(E,2) * 0.001;

% we don't use the second order stats - set 'em to empty matrix
S = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_speakers=max(spk_ids);
n_sessions=size(spk_ids,1);

% iteratively retrain v
for ii=1:niter
  disp(' ')
  disp(['Starting iteration: ' num2str(ii)])

  ct=cputime;
  et=clock;

  % go baby...
  [y v]=estimate_y_and_v(F, N, S, m, E, 0, v, 0, zeros(n_speakers,1), 0, zeros(n_sessions,1), spk_ids);

  disp(['Iteration: ' num2str(ii) ' CPU time:' num2str(cputime-ct) ' Elapsed time: ' num2str(etime(clock,et))])

  if ii == niter
	  out_final = ['output/v_final']
	  save(out_final, 'v')
  end
end

