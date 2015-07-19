/*
 * $File: pygmm.hh
 * $Date: Tue Dec 24 21:18:00 2013 +0000
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#pragma once

#include "gmm.hh"

extern "C" {
struct Parameter {
	int nr_instance;
	int nr_dim;

	int nr_mixture;

	double min_covar;
	double threshold;
	int nr_iteration;

	int init_with_kmeans;
	int concurrency;

	int verbosity;
};

GMM *new_gmm(int nr_mixture, int covariance_type);
GMM *load(const char *model_file);

void dump(GMM *gmm, const char *model_file);

void train_model(GMM *gmm, double **X_in, Parameter *param);
void train_model_from_ubm(GMM *gmm, GMM *ubm, double **X_in, Parameter *param);

double score_all(GMM *gmm, double **X_in, int nr_instance, int nr_dim, int concurrency);
void score_batch(GMM *gmm, double **X_in, double *prob_out, int nr_instance, int nr_dim, int concurrency);
double score_instance(GMM *gmm, double *x_in, int nr_dim);

int get_dim(GMM *gmm);
int get_nr_mixtures(GMM *gmm);

}

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

