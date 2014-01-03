/*
 * $File: rbm.hh
 * $Date: Thu Nov 07 12:30:21 2013 +0000
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#pragma once

#include <vector>

#include "type.hh"
#include "random.hh"

/// Restricted Boltzmann Machine
class RBM {
	public:
		// the visible_layer_size must match input dimension
		RBM(int visible_layer_size = 10, int hidden_layer_size = 10,
				int CD_k = 1, real_t learning_rate = 0.01,
				int batch_train_size = 100,
				bool hidden_use_probability = false,
				int n_iter_max = 20) {
			set_size(visible_layer_size, hidden_layer_size);
			this->learning_rate = learning_rate;
			this->CD_k = CD_k;
			this->hidden_use_probability = hidden_use_probability;
			this->batch_train_size = batch_train_size;
			this->n_iter_max = n_iter_max;
		}
		void set_size(int visible_layer_size, int hidden_layer_size);
		void set_CD_k(int CD_k) { this->CD_k = CD_k; }
		void set_learning_rate(real_t learning_rate) { this->learning_rate = learning_rate; }
		void set_n_iter_max(int n_iter_max) { this->n_iter_max = n_iter_max; }
		void set_batch_train_size(int batch_train_size) { this->batch_train_size = batch_train_size; }
		void dump(const char *fname);
		void load(const char *fname);

		/// currently, inputs are confined to binary.
		/// if real-valued inputs is given, a 0.5-threshold
		/// filter will be applied first.
		/// assuming all inputs are of the same distribution
		void fit(std::vector<std::vector<real_t>> &v);

		// fit one v at a time. this usually not workingc
		void fit_incremental(std::vector<real_t> &v);

		// begin and end are modulod by X.size()
		// if begin is greater than end, cycle from begin until reach end
		void fit_batch(std::vector<std::vector<real_t>> &X, int begin = 0, int end = 1);

		void reconstruct(std::vector<real_t> &v_in,
				std::vector<real_t> &v_out, int n_times = 1);
		void reconstruct_light(std::vector<real_t> &v_in,
				std::vector<real_t> &v_out, int n_times = 1);

		/// temporal array to store sample of
		/// hidden layer
		std::vector<real_t> h;

		std::vector<real_t> visible_layer_bias;
		std::vector<real_t> hidden_layer_bias;
		std::vector<std::vector<real_t>> w; // w[v][h]

		int get_hidden_layer_size() const { return hidden_layer_size; }
		int get_visible_layer_size() const { return visible_layer_size; }
		int get_batch_training_size() const { return batch_train_size; }

		void sample_hidden_layer(std::vector<real_t> &v,
				std::vector<real_t> &h, std::vector<real_t> &p);
		void sample_visible_layer(std::vector<real_t> &v,
				std::vector<real_t> &h);

	protected:
		real_t learning_rate;
		bool hidden_use_probability;
		int n_iter_max;
		int batch_train_size;
		int CD_k;
		int visible_layer_size, hidden_layer_size;
		Random random;

		/// seperate reconstruct purpose layer for clarity
		std::vector<real_t> h_reconstruct;

		/// temporary variable
		std::vector<std::vector<real_t>> w_0, w_inf;
		std::vector<real_t> v_0, v_inf;
		std::vector<real_t> h_0, h_inf;

		void reset_weights();
		void resize_variables();
		void resize_w(std::vector<std::vector<real_t>> &w);
		void update_parameters(std::vector<real_t> &v_0,
				std::vector<real_t> &h_0,
				std::vector<real_t> &v,
				std::vector<real_t> &h);
		void fit_batch_single(std::vector<real_t> &x);
};

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

