/*
 * $File: gmmubm.cc
 * $Date: Wed Dec 25 00:32:06 2013 +0000
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#include "gmm.hh"
#include "Threadpool/Threadpool.hpp"

#include "util.hh"

using namespace std;
using namespace ThreadLib;



GMMUBMTrainerBaseline::GMMUBMTrainerBaseline(GMM *ubm, int nr_iter, real_t min_covar,
		real_t threshold, int concurrency, int verbosity)
	: GMMTrainerBaseline(nr_iter, min_covar, threshold, 0,
			concurrency, verbosity), ubm(ubm)
{

}

void GMMUBMTrainerBaseline::init_gaussians(std::vector<std::vector<real_t>> &) {
	gmm_replace_with(gmm, ubm);
}

void GMMUBMTrainerBaseline::gmm_replace_with(GMM *gmm, GMM *ubm) {
	*gmm = *ubm;
	gmm->trainer = NULL;
	for (size_t i = 0; i < ubm->gaussians.size(); i ++) {
		// parameter initializing the Gaussian is of no importance
		// as it will be overwritten.
		gmm->gaussians[i] = new Gaussian(0);
		*gmm->gaussians[i] = *ubm->gaussians[i];
	}
}

void GMMUBMTrainerBaseline::update_weights(std::vector<std::vector<real_t>> &) {
	// do not update weights
	return;
	real_t w_sum = 0;
	for (int i = 0; i < gmm->nr_mixtures; i ++) {
		real_t alpha_i_w = N_k[i] / (N_k[i] + relevance_factor_w);
		gmm->weights[i] = alpha_i_w * N_k[i] / n + (1 - alpha_i_w) * ubm->weights[i];
		w_sum += gmm->weights[i];
	}
	for (int i = 0; i < gmm->nr_mixtures; i ++)
		gmm->weights[i] /= w_sum;
}

void GMMUBMTrainerBaseline::update_means(std::vector<std::vector<real_t>> &X) {
	Threadpool pool(concurrency);
	for (int k = 0; k < gmm->nr_mixtures; k ++) {
		auto task = [&](int k) {
			real_t alpha_i_m = N_k[k] / (N_k[k] + relevance_factor_m);
			vector<real_t> tmp(dim);
			auto &gaussian = gmm->gaussians[k];
			for (auto &v: gaussian->mean)
				v = 0;
			for (int i = 0; i < n; i ++) {
				mult(X[i], prob_of_y_given_x[k][i], tmp);
				add_self(gaussian->mean, tmp);
			}
			mult_self(gaussian->mean, 1.0 / N_k[k] * alpha_i_m); // alpha_i_m * E_i(x)

			mult(ubm->gaussians[k]->mean, 1 - alpha_i_m, tmp);
			add_self(gaussian->mean, tmp);
		};
//        task(k);
		pool.enqueue(bind(task, k), 1);
	}
}

void GMMUBMTrainerBaseline::update_variance(std::vector<std::vector<real_t>> &) {
	// do not update variance
	return;

	throw "Unimplemented";
}

/**
 * vim: syntax=cpp11.doxygen foldmethod=marker
 */

