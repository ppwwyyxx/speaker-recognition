/*
 * $File: crbm.hh
 * $Date: Fri Dec 06 21:52:23 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#pragma once

#include <vector>

#include "type.hh"
#include "random.hh"
#include <string>

class CRBMTrainer;

/// Continuous Restricted Boltzmann Machine
class CRBM {
	public:
		// the visible_layer_size must match input dimension
		CRBM(int hidden_layer_size = 10, CRBMTrainer *trainer = NULL);
		void dump(const char *fname);
		void load(const char *fname);

		std::vector<real_t> visible_layer_bias;
		std::vector<real_t> hidden_layer_bias;
		std::vector<std::vector<real_t>> w; // w[v][h]
		std::vector<real_t> a_hidden;

		std::vector<std::pair<real_t, real_t>> visible_layer_range;
		std::vector<std::pair<real_t, real_t>> hidden_layer_range;

		real_t sigma; // predefined value

		void sample_hidden_layer(std::vector<real_t> &v,
				std::vector<real_t> &h);
		void sample_visible_layer(std::vector<real_t> &v,
				std::vector<real_t> &h);

		void reconstruct(std::vector<real_t> &v_in,
				std::vector<real_t> &v_out, int n_times = 1);

		void fit(std::vector<std::vector<real_t>> &X, CRBMTrainer *trainer);
	public:
		int visible_layer_size, hidden_layer_size;
		Random random;
		CRBMTrainer *trainer;
		bool trained;
};

class CRBMTrainer {
	public:
		real_t learning_rate;
		int nr_epoch_max;
		int batch_train_size;
		int CD_k;
		bool verbose;

		int nr_epoch_report;
		int nr_reconstruction_test;
		std::string reconstruction_output_file;

		int nr_epoch_save;
		std::string model_file;


		CRBMTrainer(real_t learning_rate = 0.1, int nr_epoch_max = 100,
				int batch_train_size = 100, int CD_k = 1,
				bool verbose = true) :
			learning_rate(learning_rate),
			nr_epoch_max(nr_epoch_max),
			batch_train_size(batch_train_size),
			CD_k(CD_k),
			verbose(verbose),
			nr_epoch_report(0),
			nr_reconstruction_test(0),
			nr_epoch_save(0){}
		void train(CRBM *rbm, std::vector<std::vector<real_t>> &X);
		void train_batch(CRBM *rbm, std::vector<std::vector<real_t>> &X, int begin, int end,
				bool reset_parameters = false, bool update_coord_range = false);

	protected:
		CRBM *rbm;

		void set_variable_dimension(std::vector<std::vector<real_t>> &X);
		void train_batch_single(CRBM *rbm, std::vector<real_t> &x);

		std::vector<real_t> h; // temporal variable
		std::vector<real_t> h_0, h_inf; // temporal variable
		std::vector<real_t> v_0, v_inf; // temporal variable
		std::vector<real_t> h2_0, h2_inf; // temporal variable
		std::vector<std::vector<real_t>> w_0, w_inf; // temporal variable

		void resize_variables();
		void resize_w(std::vector<std::vector<real_t>> &w);
		void reset_parameters();
		void update_visible_coord_range(CRBM *rbm, std::vector<std::vector<real_t>> &X, int begin, int end);
		void update_visible_coord_range_single(CRBM *rbm, std::vector<real_t> &x);

		void sample_hidden_layer(std::vector<real_t> &v,
				std::vector<real_t> &h);
		void sample_visible_layer(std::vector<real_t> &v,
				std::vector<real_t> &h);

		Random random;
};


/**
 * vim: syntax=cpp11 foldmethod=marker
 */

