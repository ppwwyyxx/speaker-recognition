m             = load('models/ubm_means');
E             = load('models/ubm_variances');

trn               = load('output/enroll.mat');
trn.spk_ids       = (1:size(trn.N,1))';			% 1 ~ 20


n_speakers=max(trn.spk_ids);	% 20
n_sessions=size(trn.spk_ids,1); % 20

v_matrix_file = 'output/v_final.mat';
u_matrix_file = 'output/u_final.mat';
d_matrix_file = 'output/d_final.mat';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOAD THE FACTOR LOADING MATRICES
disp(['Loading eigenvoices matrix V from ' v_matrix_file]);
load(v_matrix_file);
ny = size(v, 1);		% 4

disp(['Loading eigenchannels matrix U from ' u_matrix_file]);
load(u_matrix_file);
nx = size(u, 1);		% 2

% we don't use the D matrix
% d = zeros(1, size(trn.F, 2));
disp(['Loading residual variability matrix D from ' d_matrix_file]);
load(d_matrix_file);


% we will estimate the channel and speaker shifts jointly, so we join the
% matrices as well
vu = [v; u];

% initialize the factors
yx      = zeros(n_speakers,ny+nx);
trn.z   = zeros(n_speakers,size(trn.F,2));
junk    = zeros(n_sessions,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENROLLMENT PART
disp('Computing factors x and y (jointly) for the enrolment stats');
% compute the joint channel and spk shift
yx = estimate_y_and_v(trn.F, trn.N, 0, m, E, d, vu, 0, trn.z, yx, junk, trn.spk_ids);

% extract the spk shift only --- we don't need the channel here
trn.y = yx(:, 1:ny);

% estimate residual variability factors z
disp('Computing factors z for the enrolment stats');
trn.z = estimate_z_and_d(trn.F, trn.N, 0, m, E, d, vu, 0, trn.z, yx, junk, trn.spk_ids);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SCORING PART
tst=load('output/test.mat');

tst.spk_ids = (1:size(tst.N,1))'

n_speakers=max(tst.spk_ids);
n_sessions=size(tst.spk_ids,1);

tst.z=zeros(n_speakers,size(tst.F,2));
tst.y=zeros(n_speakers,ny);
tst.x=zeros(n_sessions,nx);

% Now we show two options how to score:
% 1) Use linear common channel point estimate scoring
% estimate the utterance's channels wrt the UBM (LPT assumption --- all models'
% shifts are equal to UBM shift
disp('Computing factors x for the test stats');
tst.x = estimate_x_and_u(tst.F, tst.N, 0, m, E, d, v, u, tst.z, tst.y, tst.x, tst.spk_ids);

% compute the matrix of scores
disp('Calculating the score matrix');
scores = linear_scoring(tst.F, tst.N, 0, m, E, d, v, u, trn.z, trn.y, tst.x, 0);

% 2) Use scoring with integration over the whole channel distribution
% the score_mask is just a mask of which pairs to score (note that this method
% is computationaly intensive therefore we need to say explicitly which pairs
% to score for large problems.) in our demo, we just set it to ones.
% note also, that the function taks the transposed parameters (optimization
% reasons)
%%  score_mask = ones(size(trn.F,1), size(tst.F,1));
%%  scores_int = kscore_famous_19(tst.F', tst.N', 0, m', E', d', v', u', trn.z', trn.y', tst.x', score_mask);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SAVE THE BABY
disp('Saving the score matrix to exp/scores.txt');
save('exp/scores.txt', '-ascii', 'scores');

% save the labels (for simpler postprocessing)
f = fopen('exp/scores_enroll_labels.txt', 'w');
for mdl = 1:size(scores,1)
  fprintf(f, '%s\n', trn.spk_logical{mdl});
end
fclose(f);

f = fopen('exp/scores_test_labels.txt', 'w');
for utt = 1:size(scores,2)
  fprintf(f, '%s\n', tst.spk_logical{utt});
end
fclose(f);



