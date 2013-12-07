% make_suf_stat(scp_script,out_dir)
% Makes sufficient statistic for FA
%     scp_script input files
%     out_dir output directory


% load model params (as supervectors)
m=load('models/ubm_means')';
v=load('models/ubm_variances')';
w=load('models/ubm_weights')';
size_m = size(m)		% 19968
size_v = size(v)		% 19968
size_w = size(w)		% 512

n_mixtures  = size(w, 1)
dim         = size(m, 1) / n_mixtures

% we load the model as superverctors, so we reshape it to have each gaussian in
% one column
m = reshape(m, dim, n_mixtures);
v = reshape(v, dim, n_mixtures);

% these are the sets that we want to extract the stats for
data_sets{1,1} = 'enroll';
data_sets{2,1} = 'train';
data_sets{3,1} = 'test';

% process each dataset
for set_i = 1:size(data_sets,1)
  set_list_file = [ 'feature-data/' data_sets{set_i,1} '.lst' ];
  disp(['Processing list ' set_list_file]);

  % process the file list (logical=physical)
  [spk_logical spk_physical] = parse_list(set_list_file);

  n_sessions = size(spk_logical, 1);		% line number in list file

  % initialize the matrices for the stats
  % one row per session
  N = zeros(n_sessions, n_mixtures);
  F = zeros(n_sessions, n_mixtures * dim);

  % process each session
  for session_i = 1:n_sessions
    session_name = [ spk_physical{session_i,1} ];

    disp(['Reading feature file ' session_name]);
    data = load(session_name, '-ascii')';

    disp('Processing...');

    % process the feature file
    tic
    [Ni Fi] = collect_suf_stats(data, m, v, w);
    toc

    N(session_i,:) = Ni;
    F(session_i,:) = Fi;
  end

  out_stats_file = [ 'output/' data_sets{set_i,1} '.mat' ];
  disp(['Saving stats to ' out_stats_file]);
  save(out_stats_file, 'N', 'F', 'spk_logical');
  % spk_logical: speaker names
end





