/*
 * $File: crbm.cc
 * $Date: Sat Dec 07 14:26:10 2013 +0000
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#include "crbm.hh"

#include <cassert>
#include <limits>
#include <algorithm>

using namespace std;

#define RBMException(...) do { \
		printf("Exception in file %s:%d, %s: ", \
			   __FILE__, __LINE__, __PRETTY_FUNCTION__); \
		static char msg[1024]; \
		sprintf(msg, __VA_ARGS__); \
		printf("%s", msg); \
		printf("\n"); \
		throw msg; \
	} while (0)


static inline real_t sigmoid(real_t x) { return 1.0 / (1.0 + exp(-x)); }
static inline real_t sigmoid(real_t x, const std::pair<real_t, real_t> &range) {
	return range.first + (range.second - range.first) / (1.0 + exp(-x));
}


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

static void add_vec_squared(std::vector<real_t> &a,
		std::vector<real_t> &b) {
	assert(a.size() == b.size());
	for (size_t i = 0; i < a.size(); i ++)
		a[i] += b[i] * b[i];
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

static void _sample_hidden_layer(CRBM *rbm, std::vector<real_t> &v,
		std::vector<real_t> &h, Random &random, bool no_random = false) {
	for (size_t j = 0; j < h.size(); j ++) {
		real_t activation = rbm->hidden_layer_bias[j];
		for (size_t i = 0; i < v.size(); i ++)
			activation += v[i] * rbm->w[i][j];
		if (!no_random)
			activation += rbm->sigma * random.rand_normal();
		activation *= rbm->a_hidden[j];
		h[j] = sigmoid(activation, rbm->hidden_layer_range[j]);
	}
}

static void _sample_visible_layer(CRBM *rbm, std::vector<real_t> &v,
		std::vector<real_t> &h, Random &random, bool no_random = false) {
	for (size_t i = 0; i < v.size(); i ++) {
		real_t activation = rbm->visible_layer_bias[i];
		for (size_t j = 0; j < h.size(); j ++)
			activation += h[j] * rbm->w[i][j];
		if (!no_random)
			activation += rbm->sigma * random.rand_normal();
		// no 'a' factor here compared to sample_hidden_layer
		v[i] = sigmoid(activation, rbm->visible_layer_range[i]);
	}
}

CRBM::CRBM(int hidden_layer_size, CRBMTrainer *trainer) {
	this->trainer = trainer;
	this->hidden_layer_size = hidden_layer_size;
	this->visible_layer_size = -1;
	this->trained = false;
	this->sigma = 0.2;
}

void CRBM::fit(std::vector<std::vector<real_t>> &X, CRBMTrainer *trainer = NULL) {
	bool new_trainer = false;
	if (trainer == NULL) {
		if (this->trainer)
			trainer = this->trainer;
		else {
			trainer = new CRBMTrainer();
			new_trainer = true;
		}
	}
	trainer->train(this, X);
	if (new_trainer)
		delete trainer;
}

void CRBM::sample_hidden_layer(std::vector<real_t> &v,
		std::vector<real_t> &h) {
	_sample_hidden_layer(this, v, h, random);
}

void CRBM::sample_visible_layer(std::vector<real_t> &v,
		std::vector<real_t> &h) {
	_sample_visible_layer(this, v, h, random);
}


void CRBM::reconstruct(std::vector<real_t> &v_in,
		std::vector<real_t> &v_out, int n_times) {
	assert(trained == true);
	assert(v_in.size() == visible_layer_bias.size());

	std::vector<real_t> v = v_in;

	v_out.resize(v_in.size());
	std::vector<real_t> h_reconstruct;
	h_reconstruct.resize(hidden_layer_size);
	std::vector<real_t> v_sum(v_in.size(), 0);

	std::vector<real_t> p(hidden_layer_size);

	for (int time = 0; time < n_times; time ++) {
		sample_hidden_layer(v, h_reconstruct);
		sample_visible_layer(v, h_reconstruct);
	}
	v_out = v;
}

real_t CRBM::reconstruct_log_likelihood(std::vector<std::vector<real_t>> &X) {
	real_t p = 0;
	for (auto x: X)
		p += reconstruct_log_likelihood(x);
	return p;
}

real_t CRBM::reconstruction_error(std::vector<std::vector<real_t>> &X) {
	real_t error = 0;
	for (auto x: X)
		error += reconstruction_error(x);
	return error;
}

real_t CRBM::reconstruction_error(std::vector<real_t> &v_in) {
	std::vector<real_t>
		v = v_in,
		h(hidden_layer_size);
	_sample_hidden_layer(this, v, h, random, true);
	_sample_visible_layer(this, v, h, random, true);

	real_t error = 0;
	for (size_t i = 0; i < v.size(); i ++) {
		real_t e = v_in[i] - v[i];
		error += e * e;
	}
	return error;
}


real_t CRBM::reconstruct_log_likelihood(std::vector<real_t> &v_in) {
	std::vector<real_t>
		v = v_in,
		h(hidden_layer_size);
	sample_hidden_layer(v, h);
	sample_visible_layer(v, h);
	real_t energy = 0;
	for (int i = 0; i < visible_layer_size; i ++)
		for (int j = 0; j < hidden_layer_size; j ++) {
			if (i == j)
				continue;
			energy += -0.5 * w[i][j] * v[i] * h[j];
			if (fpclassify(energy) == FP_NAN)
				int asdf = 0;
		}

//    for (int i = 0; i < visible_layer_size; i ++) {
//        real_t e = (v[i] - visible_layer_bias[i]);
//        energy += e * e;
//    }

//    for (int j = 0; j < hidden_layer_size; j ++) {
//        real_t e = (h[j] - hidden_layer_bias[j]) / (sigma * sigma);
//        energy += e * e;
//    }

	for (int j = 0; j < hidden_layer_size; j ++) {
		energy += h[j] * h[j] / a_hidden[j]; // approximation
		if (fpclassify(energy) == FP_NAN)
			int asdf = 0;
	}

	return energy;
}

void CRBM::dump(const char *fname) {
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
	for (auto &a: a_hidden)
		fprintf(fout, "%.10f ", a);
	fprintf(fout, "\n");
	for (auto &r: visible_layer_range)
		fprintf(fout, "%.10f %.10f ", r.first, r.second);
	fprintf(fout, "\n");
	for (auto &r: hidden_layer_range)
		fprintf(fout, "%.10f %.10f ", r.first, r.second);
	fprintf(fout, "\n");


	fprintf(fout, "%.10f\n", sigma);
	fclose(fout);
}

void CRBM::load(const char *fname) {
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
	a_hidden.resize(hidden_layer_size);
	visible_layer_range.resize(visible_layer_size);
	hidden_layer_range.resize(hidden_layer_size);
	for (auto &a: a_hidden)
		fscanf(fin, "%lf", &a);
	for (auto &r: visible_layer_range)
		fscanf(fin, "%lf %lf", &r.first, &r.second);
	for (auto &r: hidden_layer_range)
		fscanf(fin, "%lf %lf", &r.first, &r.second);

	fscanf(fin, "%lf", &sigma);
	fclose(fin);
}


static double get_reconstruction_error(CRBM *rbm, vector<vector<real_t>> &X, int nr_reconstruction_test,
		const std::string &reconstruction_output_file) {
	FILE *fout = NULL;
	if (reconstruction_output_file.size())
		fout = fopen(reconstruction_output_file.c_str(), "w");
	double error = 0;
	vector<real_t> x;
	int nr_test = 0;
	for (int i = 0; i < (int)X.size() && i < nr_reconstruction_test; i ++, nr_test ++) {
		nr_test ++;
		vector<real_t> &v = X[i];
		rbm->reconstruct(v, x);
		if (fout) {
			for (int i = 0; i < (int)x.size(); i ++)
				fprintf(fout, "%f ", x[i]);
			fprintf(fout, "\n");
		}
		for (size_t j = 0; j < v.size(); j ++) {
			real_t e = v[j] - x[j];
			error += e * e;
		}
	}
	if (fout)
		fclose(fout);
	return error / nr_test;
}

#include <queue>
using namespace std;

// TODO: not in use currently
struct ParameterCoordinator {
	real_t cost;
	real_t prev_cost;
	real_t *learning_rate;
	real_t *momentum;
	real_t damping_factor;

	ParameterCoordinator(real_t *learning_rate = NULL,
			real_t *momentum = NULL ,
			real_t damping_factor= 0.9) :
		cost(0), prev_cost(numeric_limits<real_t>::max()),
		learning_rate(learning_rate),
		momentum(momentum),
		damping_factor(damping_factor){}

	void advance(real_t c) {
		cost = cost * (1 - damping_factor) + c * damping_factor;
	}
};

void CRBMTrainer::train(CRBM *rbm, std::vector<std::vector<real_t>> &X) {
	if (X.size() == 0)
		return;
	this->rbm = rbm;
	set_variable_dimension(X);
	resize_variables();
	reset_parameters();
	update_visible_coord_range(rbm, X, 0, X.size());

//    ParameterCoordinator pc(&learning_rate, &momentum);
	int nr_instances = (int)X.size();
	for (int epoch = 0; epoch < nr_epoch_max; epoch ++) {
		if (verbose && nr_epoch_report > 0 && epoch % nr_epoch_report == 0) {
			printf("epoch %d/%d %.4f ... ", epoch, nr_epoch_max, epoch / (double)nr_epoch_max);
			fflush(stdout);
		}
		for (int i = 0; i < nr_instances; i += batch_train_size)
			train_batch(rbm, X, i, i + batch_train_size, false, true);
		if (verbose && nr_epoch_report > 0 && epoch % nr_epoch_report == 0) {
			real_t error = get_reconstruction_error(rbm, X, nr_reconstruction_test, reconstruction_output_file);
			printf("reconstruction error: %lf\n", error);
			printf("log likelihood: %lf\n", rbm->reconstruct_log_likelihood(X));
//            pc.advance(error);
//            printf("cost: %lf\n", pc.cost);
			fflush(stdout);
		}
		if (verbose && nr_epoch_save > 0 && epoch % nr_epoch_save == 0) {
			printf("saving model to `%s' ...\n", model_file.c_str());
			if (model_file.size() != 0)
				rbm->dump(model_file.c_str());
			else printf("model file not specified, abort saving.\n");
			fflush(stdout);
		}
	}
}

void CRBMTrainer::set_variable_dimension(std::vector<std::vector<real_t>> &X) {
	if (X.size() == 0)
		RBMException("size of X is 0.");
	if (this->rbm->visible_layer_size == -1)
		this->rbm->visible_layer_size = X[0].size();
	else if (this->rbm->visible_layer_size != (int)X[0].size())
		RBMException("data dimension %d does not consist with visible_layer_size %d",
				(int)X[0].size(), this->rbm->visible_layer_size);
}

void CRBMTrainer::train_batch(CRBM *rbm, std::vector<std::vector<real_t>> &X, int begin, int end,
		bool reset_parameters, bool update_coord_range) {
	this->rbm = rbm;
	set_variable_dimension(X);
	resize_variables();
	if (!rbm->trained || reset_parameters) {
		this->reset_parameters(); // do not reset weights if this rbm has been trained before
	}
	if (update_coord_range)
		this->update_visible_coord_range(rbm, X, begin, end);
	clear_vec(v_0); clear_vec(v_inf);
	clear_vec(h_0); clear_vec(h_inf);
	clear_vec(h2_0); clear_vec(h2_inf);
	clear_w(w_0); clear_w(w_inf);

	int n_instance = 0;

	for (int i = begin; i != end; i ++) {
		n_instance ++;
		train_batch_single(rbm, X[i % X.size()]);
	}

	divide_vec(v_0, n_instance); divide_vec(v_inf, n_instance);
	divide_vec(h_0, n_instance); divide_vec(h_inf, n_instance);
	divide_vec(h2_0, n_instance); divide_vec(h2_inf, n_instance);
	divide_w(w_0, n_instance); divide_w(w_inf, n_instance);

	// update weights
	for (size_t i = 0; i < rbm->visible_layer_bias.size(); i ++) {
		real_t delta_v = (v_0[i] - v_inf[i]);
		//printf("dv[%d] = %f\n", (int)i, delta_v);
		dv[i] = momentum * dv[i] + learning_rate * (delta_v - C * rbm->visible_layer_bias[i]);
		rbm->visible_layer_bias[i] += dv[i];
//        rbm->visible_layer_bias[i] += learning_rate * (delta_v - C * rbm->visible_layer_bias[i]);
	}
	for (size_t j = 0; j < rbm->hidden_layer_bias.size(); j ++) {
		real_t delta_h = (h_0[j] - h_inf[j]);
//        printf("dh[%d] = %f\n", (int)j, delta_h);
        dh[j] = momentum * dh[j] + learning_rate * (delta_h - C * rbm->hidden_layer_bias[j]);
		rbm->hidden_layer_bias[j] += dh[j];
//        rbm->hidden_layer_bias[j] += learning_rate * (delta_h - C * rbm->hidden_layer_bias[j]);
	}
	for (size_t i = 0; i < rbm->visible_layer_bias.size(); i ++)
		for (size_t j = 0; j < rbm->hidden_layer_bias.size(); j ++)
		{
			real_t delta_w = (w_0[i][j] - w_inf[i][j]);
			dw[i][j] = momentum * dw[i][j] + learning_rate * (delta_w - C * rbm->w[i][j]);
			//printf("dw[%d][%d] = %f\n", (int)i, (int)j, delta_w);
			rbm->w[i][j] += dw[i][j];
		}
	for (int i = 0; i < rbm->hidden_layer_size; i ++){
		real_t ah_i = rbm->a_hidden[i];
		real_t delta_a = (h2_0[i] - h2_inf[i]) / (ah_i * ah_i);
		da_hid[i] = momentum * da_hid[i] + learning_rate * (delta_a - C * rbm->a_hidden[i]);
		rbm->a_hidden[i] += da_hid[i];
	}

	rbm->trained = true;
}

void CRBMTrainer::train_batch_single(CRBM *rbm, std::vector<real_t> &x) {
	// make a copy
	std::vector<real_t> v = x;

	std::vector<real_t> p(rbm->hidden_layer_size);
	sample_hidden_layer(v, h);
	add_vec(v_0, v); add_vec(h_0, h);
	add_vec_squared(h2_0, h);
	add_w(w_0, v, h);
	for (int j = 0; j < CD_k; j ++) {
		sample_visible_layer(v, h);
		sample_hidden_layer(v, h);
	}
	add_vec(v_inf, v); add_vec(h_inf, h);
	add_vec_squared(h2_inf, h);
	add_w(w_inf, v, h);
}

void CRBMTrainer::reset_parameters() {
	for (auto &vb: rbm->visible_layer_bias)
		vb = random.rand_normal() * 0.01;
	for (auto &hb: rbm->hidden_layer_bias)
		hb = random.rand_normal() * 0.01;
	for (auto &visible: rbm->w)
		for (auto &val: visible)
			val = random.rand_normal() * 0.01;
	for (auto &a: rbm->a_hidden)
		a = 1.0;

	for (auto &v: dh) v = 0;
	for (auto &v: dv) v = 0;
	for (auto &v: da_hid) v = 0;
	for (auto &v: dw)
		for (auto &h: v)
			h = 0;

	rbm->hidden_layer_range = vector<pair<real_t,real_t>>(rbm->hidden_layer_size,
			make_pair(0.0, 1.0));
	real_t inf = numeric_limits<real_t>::max();
	rbm->visible_layer_range = vector<pair<real_t,real_t>>(rbm->visible_layer_size,
			make_pair(inf, -inf));
}

void CRBMTrainer::resize_variables() {
	rbm->visible_layer_bias.resize(rbm->visible_layer_size);
	rbm->hidden_layer_bias.resize(rbm->hidden_layer_size);
	h.resize(rbm->hidden_layer_size);
	resize_w(rbm->w);
	rbm->a_hidden.resize(rbm->hidden_layer_size);

	h_0.resize(rbm->hidden_layer_size);
	h_inf.resize(rbm->hidden_layer_size);
	v_0.resize(rbm->visible_layer_size);
	v_inf.resize(rbm->visible_layer_size);
	resize_w(w_0);
	resize_w(w_inf);
	h2_0.resize(rbm->hidden_layer_size);
	h2_inf.resize(rbm->hidden_layer_size);

	dh.resize(rbm->hidden_layer_size);
	dv.resize(rbm->visible_layer_size);
	da_hid.resize(rbm->hidden_layer_size);
	resize_w(dw);
}

void CRBMTrainer::resize_w(std::vector<std::vector<real_t>> &w) {
	w.resize(rbm->visible_layer_size);
	for (auto &h: w)
		h.resize(rbm->hidden_layer_size);
}

void CRBMTrainer::update_visible_coord_range_single(CRBM *rbm, std::vector<real_t> &x) {
	assert((int)x.size() == rbm->visible_layer_size);
	for (size_t i = 0; i < x.size(); i ++) {
		auto &range = rbm->visible_layer_range[i];
		if (x[i] < range.first)
			range.first = x[i];
		if (x[i] > range.second)
			range.second = x[i];
	}
}

void CRBMTrainer::update_visible_coord_range(CRBM *rbm, std::vector<std::vector<real_t>> &X,
		int begin, int end) {
	int start_ind = begin % X.size();
	update_visible_coord_range_single(rbm, X[start_ind]);
	for (int i = begin + 1; i < end; i ++) {
		int ind = i % X.size();
		if (ind == start_ind)
			break;
		update_visible_coord_range_single(rbm, X[ind]);
	}
}

void CRBMTrainer::sample_hidden_layer(std::vector<real_t> &v,
		std::vector<real_t> &h) {
	_sample_hidden_layer(rbm, v, h, random);
}

void CRBMTrainer::sample_visible_layer(std::vector<real_t> &v,
		std::vector<real_t> &h) {
	_sample_visible_layer(rbm, v, h, random);
}

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

