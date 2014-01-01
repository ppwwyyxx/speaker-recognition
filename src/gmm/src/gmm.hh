/*
 * $File: gmm.hh
 * $Date: Wed Dec 25 01:33:16 2013 +0000
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#pragma once

#include "type.hh"
#include "random.hh"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>


enum CovType {
	COVTYPE_SPHERICAL,
	COVTYPE_DIAGONAL,
	COVTYPE_FULL
};

class Gaussian {
	public:
		Gaussian(int dim, int covariance_type = COVTYPE_DIAGONAL);

		int dim, covariance_type;
		std::vector<real_t> mean;
		std::vector<real_t> sigma;
		std::vector<std::vector<real_t>> covariance; // not used

		real_t log_probability_of(std::vector<real_t> &x);
		real_t probability_of(std::vector<real_t> &x);
		real_t probability_of_fast_exp(std::vector<real_t> &x, double *buffer = NULL);

		// sample a point to @x
		void sample(std::vector<real_t> &x);
		std::vector<real_t> sample();

		void dump(std::ostream &out);
		void load(std::istream &in);

		Random random;
		int fast_gaussian_dim;
};

class GMM;
class GMMTrainer {
	public:
		virtual void train(GMM *gmm, std::vector<std::vector<real_t>> &X) = 0;
		virtual ~GMMTrainer() {}
};

class GMMTrainerBaseline : public GMMTrainer {
	public:
		GMMTrainerBaseline(int nr_iter = 10, //!< number of iteration
				real_t min_covar = 1e-3, //!< minimum covariance to avoid overfitting
				real_t threshold = 0.01, //!< threshold to stop iteration
				int init_with_kmeans = 1, //!< initialize using kmeans
				int concurrency = 1, //!< concurrency
				int verbosity = 0);
		~GMMTrainerBaseline();
		virtual void train(GMM *gmm, std::vector<std::vector<real_t>> &X);
		void clear_gaussians();

		// init gaussians along with its weight
		virtual void init_gaussians(std::vector<std::vector<real_t>> &X);

		virtual void iteration(std::vector<std::vector<real_t>> &X);

		virtual void update_weights(std::vector<std::vector<real_t>> &X);
		virtual void update_means(std::vector<std::vector<real_t>> &X);
		virtual void update_variance(std::vector<std::vector<real_t>> &X);

		int dim;

		Random random;
		GMM *gmm;

		int nr_iter;
		real_t min_covar;
		real_t threshold;

		int init_with_kmeans;

		int concurrency;
		int verbosity;

		std::vector<std::vector<real_t>> prob_of_y_given_x; // y, x
		std::vector<real_t> N_k;
		std::vector<real_t> prob_sum;

		int n; // training sample size
};

class GMMUBMTrainerBaseline : public GMMTrainerBaseline {
	public:
		GMMUBMTrainerBaseline(GMM *ubm, int nr_iter = 10,
				real_t min_covar = 1e-3,
				real_t threshold = 0.01,
				int concurrency = 1,
				int verbosity = 0);

		virtual void init_gaussians(std::vector<std::vector<real_t>> &X);
		virtual void update_weights(std::vector<std::vector<real_t>> &X);
		virtual void update_means(std::vector<std::vector<real_t>> &X);
		virtual void update_variance(std::vector<std::vector<real_t>> &X);

		real_t relevance_factor_w = 16; //!< relevance factor of weights
		real_t relevance_factor_m = 16; //!< relevance factor of means
		real_t relevance_factor_v = 16; //!< relevance factor of variance

		GMM *ubm;

		/*!
		 * replace parameters of gmm with ubm.
		 * the trainer of gmm is set to NULL
		 */
		void gmm_replace_with(GMM *gmm, GMM *ubm);
};

class GMM {
	public:
		GMM(int nr_mixtures, int covariance_type = COVTYPE_DIAGONAL,
				GMMTrainer *trainer = NULL);
		GMM(const std::string &model_file);
		~GMM();

		template<class Instance>
			void fit(std::vector<Instance> &X) {
				bool new_trainer = false;
				if (trainer == NULL) {
					trainer = new GMMTrainerBaseline();
					new_trainer = true;
				}
				trainer->train(this, X);
				if (new_trainer)
					delete trainer;
			}

		void dump(std::ostream &out);
		void load(std::istream &in);

		int nr_mixtures;
		int covariance_type;

		int dim;
		GMMTrainer *trainer;

		std::vector<real_t> weights;
		std::vector<Gaussian *> gaussians;

		real_t log_probability_of(std::vector<real_t> &x);
		real_t log_probability_of(std::vector<std::vector<real_t>> &X);

		real_t log_probability_of_fast_exp(std::vector<real_t> &x, double *buffer = NULL);
		real_t log_probability_of_fast_exp(std::vector<std::vector<real_t>> &X, double *buffer = NULL);
		real_t log_probability_of_fast_exp_threaded(std::vector<std::vector<real_t>> &X, int concurrency);
		void log_probability_of_fast_exp_threaded(
				std::vector<std::vector<real_t>> &X, std::vector<real_t> &prob_out, int concurrency);


		real_t probability_of(std::vector<real_t> &x);

		void normalize_weights();

};

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

