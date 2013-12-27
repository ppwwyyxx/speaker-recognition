/*
 * $File: pygmm.cc
 * $Date: Fri Dec 27 11:37:56 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#include "pygmm.hh"
#include "gmm.hh"

#include <fstream>

using namespace std;

typedef vector<vector<real_t>> DenseDataset;

void conv_double_pp_to_vv(double **Xp, DenseDataset &X, int nr_instance, int nr_dim) {
	X.resize(nr_instance);
	for (auto &x: X)
		x.resize(nr_dim);
	for (int i = 0; i < nr_instance; i ++)
		for (int j = 0; j < nr_dim; j ++)
			X[i][j] = Xp[i][j];
}

void conv_double_p_to_v(double *x_in, vector<real_t> &x, int nr_dim) {
	x.resize(nr_dim);
	for (int i = 0; i < nr_dim; i ++)
		x[i] = x_in[i];
}

void print_param(Parameter *param) {
	printf("nr_instance   :   %d\n", param->nr_instance);
	printf("nr_dim        :   %d\n", param->nr_dim);
	printf("nr_mixture    :   %d\n", param->nr_mixture);
	printf("min_covar     :   %f\n", param->min_covar);
	printf("threshold     :   %f\n", param->threshold);
	printf("nr_iteration  :   %d\n", param->nr_iteration);
	printf("init_with_kmeans: %d\n", param->init_with_kmeans);
	printf("concurrency   :   %d\n", param->concurrency);
	printf("verbosity     :   %d\n", param->verbosity);
}

void print_X(double **X) {
	printf("X: %p\n", X);
	printf("X: %p\n", X[0]);
	printf("X: %f\n", X[0][0]);
}


GMM *new_gmm(int nr_mixture, int covariance_type) {
	return new GMM(nr_mixture, covariance_type);
}

GMM *load(const char *model_file) {
	return new GMM(model_file);
}

void dump(GMM *gmm, const char *model_file) {
	ofstream fout(model_file);
	gmm->dump(fout);
}

void train_model(GMM *gmm, double **X_in, Parameter *param) {
	print_param(param);
	GMMTrainerBaseline trainer(param->nr_iteration, param->min_covar,
			param->threshold,
			param->init_with_kmeans, param->concurrency, param->verbosity);
	gmm->trainer = &trainer;
	DenseDataset X;
	conv_double_pp_to_vv(X_in, X, param->nr_instance, param->nr_dim);
	/*
	 *{
	 *    ofstream of("dump.txt");
	 *    for (auto & row : X) {
	 *        for (auto & v : row) {
	 *            of << v << " ";
	 *        }
	 *        of << endl;
	 *    }
	 *    of.close();
	 *}
	 */
	gmm->fit(X);
	gmm->trainer = NULL;
}

void train_model_from_ubm(GMM *gmm, GMM *ubm, double **X_in, Parameter *param) {
	print_param(param);
	GMMUBMTrainerBaseline trainer(ubm, param->nr_iteration, param->min_covar,
			param->threshold, param->concurrency, param->verbosity);
	gmm->trainer = &trainer;
	DenseDataset X;
	conv_double_pp_to_vv(X_in, X, param->nr_instance, param->nr_dim);
	gmm->fit(X);
	gmm->trainer = NULL;
}

double score_all(GMM *gmm, double **X_in, int nr_instance, int nr_dim, int concurrency) {
	DenseDataset X;
	conv_double_pp_to_vv(X_in, X, nr_instance, nr_dim);
	return gmm->log_probability_of_fast_exp_threaded(X, concurrency);
}

void score_batch(GMM *gmm, double **X_in, double *prob_out, int nr_instance, int nr_dim, int concurrency) {
	DenseDataset X;
	conv_double_pp_to_vv(X_in, X, nr_instance, nr_dim);
	std::vector<real_t> prob;
	gmm->log_probability_of_fast_exp_threaded(X, prob, concurrency);
	for (size_t i = 0; i < prob.size(); i ++)
		prob_out[i] = prob[i];
}

double score_instance(GMM *gmm, double *x_in, int nr_dim) {
	vector<real_t> x;
	conv_double_p_to_v(x_in, x, nr_dim);
	return gmm->log_probability_of_fast_exp(x);
}

int get_dim(GMM *gmm) { return gmm->dim; }
int get_nr_mixtures(GMM *gmm) { return gmm->nr_mixtures; }



/**
 * vim: syntax=cpp11 foldmethod=marker
 */

