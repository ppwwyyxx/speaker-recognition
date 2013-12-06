/*
 * $File: rbm.cc
 * $Date: Thu Dec 05 00:08:30 2013 +0000
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#include "rbm.hh"

#include <cassert>

using namespace std;

static inline real_t sigmoid(real_t x) { return 1.0 / (1.0 + exp(-x)); }

static void clear_vec(std::vector<real_t> &x) {
	for (auto &v: x)
		v = 0;
}

static void clear_w(std::vector<std::vector<real_t>> &w) {
	for (auto &v: w)
		for (auto &x: v)
			x = 0;
}

static void add_vec(std::vector<real_t> &a,
		std::vector<real_t> &b) {
	assert(a.size() == b.size());
	for (size_t i = 0; i < a.size(); i ++)
		a[i] += b[i];
}

static void add_w(std::vector<std::vector<real_t>> &w_a,
		std::vector<std::vector<real_t>> &w_b) {
	assert(w_a.size() == w_b.size());
	for (size_t i = 0; i < w_a.size(); i ++) {
		assert(w_a[i].size() == w_b[i].size());
		for (size_t j = 0; j < w_b.size(); j ++) {
			w_a[i][j] += w_b[i][j];
		}
	}
}

static void add_w(std::vector<std::vector<real_t>> &w,
		std::vector<real_t> &v, std::vector<real_t> &h) {
	assert(w.size() == v.size());
	for (size_t i = 0; i < v.size(); i ++) {
		assert(w[i].size() == h.size());
		real_t v_tmp = v[i];
		for (size_t j = 0; j < h.size(); j ++) {
			w[i][j] += v_tmp * h[j];
		}
	}
}

static void divide_vec(std::vector<real_t> &a, real_t b) {
	real_t b_inv = 1.0 / b;
	for (auto &v: a) v *= b_inv;
}

static void divide_w(std::vector<std::vector<real_t>> &w, real_t b) {
	real_t b_inv = 1.0 / b;
	for (auto &v: w)
		for (auto &t: v)
			t *= b_inv;
}

void RBM::fit_batch_single(std::vector<real_t> &x) {
	// make a copy
	std::vector<real_t> v = x;

	std::vector<real_t> p(hidden_layer_size);
	sample_hidden_layer(v, h, p);
	add_vec(v_0, v); add_vec(h_0, p); // NOTE: here use p for smaller noise
	add_w(w_0, v, h);
	for (int j = 0; j < CD_k; j ++) {
		sample_visible_layer(v, h);
		sample_hidden_layer(v, h, p);
	}
	add_vec(v_inf, v); add_vec(h_inf, p);
	add_w(w_inf, v, h);
}

void RBM::fit_batch(std::vector<std::vector<real_t>> &X, int begin, int end) {
	resize_variables();
	clear_vec(v_0); clear_vec(v_inf);
	clear_vec(h_0); clear_vec(h_inf);
	clear_w(w_0); clear_w(w_inf);

	int n_instance = 0;

	for (int i = begin; i != end; i ++) {
		n_instance ++;
		fit_batch_single(X[i % X.size()]);
	}

	divide_vec(v_0, n_instance); divide_vec(v_inf, n_instance);
	divide_vec(h_0, n_instance); divide_vec(h_inf, n_instance);
	divide_w(w_0, n_instance); divide_w(w_inf, n_instance);

	// update weights
	for (size_t i = 0; i < visible_layer_bias.size(); i ++) {
		real_t delta_v = (v_0[i] - v_inf[i]);
		//printf("dv[%d] = %f\n", (int)i, delta_v);
		visible_layer_bias[i] += learning_rate * delta_v;
	}
	for (size_t j = 0; j < hidden_layer_bias.size(); j ++) {
		real_t delta_h = (h_0[j] - h_inf[j]);
		//printf("dh[%d] = %f\n", (int)j, delta_h);
		hidden_layer_bias[j] += learning_rate * delta_h;
	}
	for (size_t i = 0; i < visible_layer_bias.size(); i ++)
		for (size_t j = 0; j < hidden_layer_bias.size(); j ++)
		{
			real_t delta_w = (w_0[i][j] - w_inf[i][j]);
			//printf("dw[%d][%d] = %f\n", (int)i, (int)j, delta_w);
			w[i][j] += learning_rate * delta_w;
		}
	//printf("abcde\n");
}

void RBM::update_parameters(std::vector<real_t> &v_0,
		std::vector<real_t> &h_0,
		std::vector<real_t> &v,
		std::vector<real_t> &h) {

	for (size_t i = 0; i < v.size(); i ++)
		for (size_t j = 0; j < h.size(); j ++) {
			real_t delta_w = h_0[j] * v_0[i] - h[j] * v[i];
			w[i][j] += learning_rate * delta_w;
		}
	for (size_t i = 0; i < v.size(); i ++)
		visible_layer_bias[i] += learning_rate * (v_0[i] - v[i]);
	for (size_t j = 0; j < h.size(); j ++)
		hidden_layer_bias[j] += learning_rate * (h_0[j] - h[j]);
}

void RBM::sample_hidden_layer(std::vector<real_t> &v,
		std::vector<real_t> &h, std::vector<real_t> &p) {
	for (size_t j = 0; j < h.size(); j ++) {
		real_t activation = hidden_layer_bias[j];
		for (size_t i = 0; i < v.size(); i ++)
			activation += v[i] * w[i][j];
		real_t prob = sigmoid(activation);
		p[j] = prob;
		h[j] = random.rand_real() < prob;
	}
}

void RBM::sample_visible_layer(std::vector<real_t> &v,
		std::vector<real_t> &h) {
	for (size_t i = 0; i < v.size(); i ++) {
		real_t activation = visible_layer_bias[i];
		for (size_t j = 0; j < h.size(); j ++)
			activation += h[j] * w[i][j];
		real_t prob = sigmoid(activation);
		v[i] = random.rand_real() < prob;
	}
}

void RBM::fit(std::vector<std::vector<real_t>> &X) {
	if (X.size() == 0)
		return;
	assert(X[0].size() == (size_t)visible_layer_size);

	resize_variables();
	reset_weights();

	// fit batch
	for (size_t i = 0; ; ) {
		fit_batch(X, i, i + batch_train_size);
		i += batch_train_size;
		if (i >= n_iter_max * X.size())
			break;
	}
}

void RBM::reconstruct_light(std::vector<real_t> &v_in,
		std::vector<real_t> &v_out, int n_times) {
	resize_variables();
	assert(v_in.size() == visible_layer_bias.size());

	std::vector<real_t> v = v_in;

	v_out.resize(v_in.size());
	h_reconstruct.resize(hidden_layer_size);
	std::vector<real_t> v_sum(v_in.size(), 0);

	std::vector<real_t> p(hidden_layer_size);

	for (int time = 0; time < n_times; time ++) {
		sample_hidden_layer(v, h_reconstruct, p);
		sample_visible_layer(v, h_reconstruct);
	}
	v_out = v;
}

void RBM::reconstruct(std::vector<real_t> &v_in,
		std::vector<real_t> &v_out, int n_times) {
	resize_variables();
	assert(v_in.size() == visible_layer_bias.size());
	std::vector<real_t> v = v_in;
	v_out.resize(v_in.size());
	h_reconstruct.resize(hidden_layer_size);
	std::vector<real_t> v_sum(v_in.size(), 0);

	std::vector<real_t> p(hidden_layer_size);

	for (int time = 0; time < n_times; time ++) {
		sample_hidden_layer(v, h_reconstruct, p);
		sample_visible_layer(v, h_reconstruct);
		for (size_t i = 0; i < v_in.size(); i ++)
			v_sum[i] += v[i];
	}
	for (size_t i = 0; i < v_in.size(); i ++)
		v_out[i] = (real_t)v_sum[i] / n_times;
}

void RBM::resize_variables() {
	visible_layer_bias.resize(visible_layer_size);
	hidden_layer_bias.resize(hidden_layer_size);
	h.resize(hidden_layer_size);
	resize_w(w);

	h_0.resize(hidden_layer_size);
	h_inf.resize(hidden_layer_size);
	v_0.resize(visible_layer_size);
	v_inf.resize(visible_layer_size);
	resize_w(w_0);
	resize_w(w_inf);
}

void RBM::resize_w(std::vector<std::vector<real_t>> &w) {
	w.resize(visible_layer_size);
	for (auto &h: w)
		h.resize(hidden_layer_size);
}

void RBM::reset_weights() {
	for (auto &vb: visible_layer_bias)
		vb = random.rand_normal() * 0.01;
	for (auto &hb: hidden_layer_bias)
		hb = random.rand_normal() * 0.01;
	for (auto &visible: w)
		for (auto &val: visible)
			val = random.rand_normal() * 0.01;
}

void RBM::set_size(int visible_layer_size, int hidden_layer_size) {
	this->visible_layer_size = visible_layer_size;
	this->hidden_layer_size = hidden_layer_size;
}

void RBM::dump(const char *fname) {
	FILE *fout = fopen(fname, "w");
	fprintf(fout, "%d %d\n", visible_layer_size, hidden_layer_size);
	for (auto &v: visible_layer_bias) fprintf(fout, "%.10f ", v);
	fprintf(fout, "\n");
	for (auto &v: hidden_layer_bias) fprintf(fout, "%.10f ", v);
	fprintf(fout, "\n");
	for (int i = 0; i < visible_layer_size; i ++) {
		for (int j = 0; j < hidden_layer_size; j ++)
			fprintf(fout, "%.10f ", w[i][j]);
		fprintf(fout, "\n");
	}
	fclose(fout);
}

void RBM::load(const char *fname) {
	FILE *fin = fopen(fname, "r");
	fscanf(fin, "%d%d", &visible_layer_size, &hidden_layer_size);
	visible_layer_bias.resize(visible_layer_size);
	hidden_layer_bias.resize(hidden_layer_size);
	for (auto &v: visible_layer_bias) fscanf(fin, "%lf ", &v);
	for (auto &v: hidden_layer_bias) fscanf(fin, "%lf ", &v);
	w.resize(visible_layer_size);
	for (int i = 0; i < visible_layer_size; i ++) {
		w[i].resize(hidden_layer_size);
		for (int j = 0; j < hidden_layer_size; j ++)
			fscanf(fin, "%lf", &w[i][j]);
	}
	fclose(fin);
}

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

