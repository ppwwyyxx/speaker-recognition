/*
 * $File: gmm.cc
 * $Date: Fri Dec 27 02:50:18 2013 +0000
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#include "gmm.hh"
#include "timer.hh"
#include "Threadpool/Threadpool.hpp"
#include "util.hh"

#include "kmeansII.hh"

#include <cassert>
#include <fstream>
#include <limits>


using namespace std;
using namespace ThreadLib;

static const real_t SQRT_2_PI = 2.5066282746310002;

#include "fastexp.hh"

#define array_exp remez5_0_log2_sse

#define dprintf(fmt, ...) \
	printf(fmt, ##__VA_ARGS__);


static const real_t EPS = 2.2204460492503131e-16;

static real_t safe_log(real_t x) {
	if (x <= 0)
		x = 1e-15;
	return log(x);
}

Gaussian::Gaussian(int dim, int covariance_type) :
	dim(dim), covariance_type(covariance_type) {
	if (covariance_type != COVTYPE_DIAGONAL) {
		const char *msg = "only diagonal matrix supported.";
		printf("%s\n", msg);
		throw msg;
	}
	sigma.resize(dim);
	mean.resize(dim);

	fast_gaussian_dim = (int)(ceil(dim / 4.0) * 4);
}

GMMTrainerBaseline::~GMMTrainerBaseline() {
}

void Gaussian::sample(std::vector<real_t> &x) {
	switch (covariance_type) {
		case COVTYPE_SPHERICAL:
			throw "COVTYPE_SPHERICAL not implemented";
			break;
		case COVTYPE_DIAGONAL:
			x.resize(dim);
			for (int i = 0; i < dim; i ++)
				x[i] = random.rand_normal(mean[i], sigma[i]);
			break;
		case COVTYPE_FULL:
			throw "COVTYPE_FULL not implemented";
			break;
	}
}

vector<real_t> Gaussian::sample() {
	vector<real_t> x;
	sample(x);
	return x;
}

real_t Gaussian::log_probability_of(std::vector<real_t> &x) {
	assert((int)x.size() == dim);

	real_t prob = 0;
	switch (covariance_type) {
		case COVTYPE_SPHERICAL:
			throw "COVTYPE_SPHERICAL not implemented";
			break;
		case COVTYPE_DIAGONAL:
			for (int i = 0; i < dim; i ++) {
				real_t &s = sigma[i];
				real_t s2 = s * s;
				real_t d = (x[i] - mean[i]);
				prob += -safe_log(SQRT_2_PI * s) - 1.0 / (2 * s2) * d * d;
			}
			break;
		case COVTYPE_FULL:
			throw "COVTYPE_FULL not implemented";
			break;
	}
	return prob;
}

void Gaussian::dump(std::ostream &out) {
	out << dim << ' ' << covariance_type << endl;
	for (auto &m: mean) out << m << ' ';
	out << endl;

	// output sigma
	switch (covariance_type) {
		case COVTYPE_SPHERICAL:
			throw "COVTYPE_SPHERICAL not implemented";
			break;
		case COVTYPE_DIAGONAL:
			for (auto &s: sigma) out << s << ' ';
			out << endl;
			break;
		case COVTYPE_FULL:
			for (auto &row: covariance) {
				for (auto &v: row)
					out << v << ' ';
				out << endl;
			}
			break;
	}
}

void Gaussian::load(std::istream &in) {
	in >> dim >> covariance_type;
	mean.resize(dim);
	for (auto &m: mean) in >> m;
	fast_gaussian_dim = (int)(ceil(dim / 4.0) * 4);

	// input sigma
	switch (covariance_type) {
		case COVTYPE_SPHERICAL:
			throw "COVTYPE_SPHERICAL not implemented";
			break;
		case COVTYPE_DIAGONAL:
			sigma.resize(dim);
			for (auto &s: sigma) in >> s;
			break;
		case COVTYPE_FULL:
			throw "COVTYPE_FULL not implemented";
			covariance.resize(dim);
			for (auto &row: covariance) {
				row.resize(dim);
				for (auto &v: row)
					in >> v;
			}
			break;
	}
}

// most time consuming function
real_t Gaussian::probability_of(std::vector<real_t> &x) {
	assert((int)x.size() == dim);

	real_t prob = 1.0;
	switch (covariance_type) {
		case COVTYPE_SPHERICAL:
			throw "COVTYPE_SPHERICAL not implemented";
			break;
		case COVTYPE_DIAGONAL:
				for (int i = 0; i < dim; i ++) {
					real_t &s = sigma[i];
					real_t d = x[i] - mean[i];
					real_t p = exp(- d * d / (2 * s * s)) / (SQRT_2_PI * s);
					prob *= p;
				}
			break;
		case COVTYPE_FULL:
			throw "COVTYPE_FULL not implemented";
			break;
	}
	return prob;
}

real_t Gaussian::probability_of_fast_exp(std::vector<real_t> &x, double *buffer) {
	assert((int)x.size() == dim);

	real_t prob = 1.0;
	switch (covariance_type) {
		case COVTYPE_SPHERICAL:
			throw "COVTYPE_SPHERICAL not implemented";
			break;
		case COVTYPE_DIAGONAL:
			assert(buffer != NULL);
			for (int i = 0; i < dim; i ++) {
				real_t &s = sigma[i];
				real_t d = x[i] - mean[i];
				buffer[i] = - d * d / (2 * s * s);
			}
			array_exp(buffer, fast_gaussian_dim);
			for (int i = 0; i < dim; i ++) {
				real_t p = buffer[i] / (SQRT_2_PI * sigma[i]);
				prob *= p;
			}
			break;
		case COVTYPE_FULL:
			throw "COVTYPE_FULL not implemented";
			break;
	}
	return prob;
}


GMM::GMM(int nr_mixtures, int covariance_type,
		GMMTrainer *trainer) :
	nr_mixtures(nr_mixtures),
	covariance_type(covariance_type),
	trainer(trainer) {

	if (covariance_type != COVTYPE_DIAGONAL) {
		const char *msg = "only diagonal matrix supported.";
		printf("%s\n", msg);
		throw msg;
	}
}

GMM::GMM(const std::string &model_file) {
	ifstream fin(model_file);
	this->load(fin);
}

GMM::~GMM() {
	for (auto &g: gaussians)
		delete g;
}


real_t GMM::log_probability_of(std::vector<real_t> &x) {
	real_t prob = 0;
	for (int i = 0; i < nr_mixtures; i ++) {
		prob += weights[i] * gaussians[i]->probability_of(x);
	}
	return safe_log(prob);
}

real_t GMM::log_probability_of_fast_exp(std::vector<real_t> &x, double *buffer) {

	real_t prob = 0;
	for (int i = 0; i < nr_mixtures; i ++) {
		prob += weights[i] * gaussians[i]->probability_of_fast_exp(x, buffer);
	}
	return safe_log(prob);
}

real_t GMM::probability_of(std::vector<real_t> &x) {
	real_t prob = 0;
	for (int i = 0; i < nr_mixtures; i ++) {
		prob *= weights[i] * gaussians[i]->probability_of(x);
	}
	return prob;
}

// time consuming
real_t GMM::log_probability_of(std::vector<std::vector<real_t>> &X) {
	real_t prob = 0;
	for (auto &x: X)
		prob += log_probability_of(x);
	return prob;
}

real_t GMM::log_probability_of_fast_exp(std::vector<std::vector<real_t>> &X, double *buffer) {
	assert(buffer != NULL);
	real_t prob = 0;
	for (auto &x: X)
		prob += log_probability_of_fast_exp(x, buffer);
	return prob;
}

#if 0
static vector<real_t> random_vector(int dim, real_t range, Random &random) {
	vector<real_t> vec(dim);
	for (auto &v: vec) v = random.rand_real() * range;
	return vec;
}
#endif

GMMTrainerBaseline::GMMTrainerBaseline(int nr_iter, real_t min_covar,
		real_t threshold,
		int init_with_kmeans,
		int concurrency, int verbosity) :
	nr_iter(nr_iter), min_covar(min_covar), threshold(threshold),
	init_with_kmeans(init_with_kmeans), concurrency(concurrency),
	verbosity(verbosity) {
}


static void dense2sparse(const std::vector<real_t> &x, Instance &inst) {
	inst.resize(x.size());
	for (size_t i = 0; i < x.size(); i ++) {
		inst[i].first = i;
		inst[i].second = x[i];
	}
}

static void Dense2Sparse(const std::vector<std::vector<real_t>> &X,
		Dataset &X_sparse) {
	X_sparse.resize(0);
	for (auto &x: X) {
		Instance inst;
		dense2sparse(x, inst);
		X_sparse.push_back(inst);
	}
}

void GMMTrainerBaseline::init_gaussians(std::vector<std::vector<real_t>> &X) {
	assert(gmm->covariance_type == COVTYPE_DIAGONAL);

	// calculate data variance
	vector<real_t> initial_sigma(dim);
	vector<real_t> data_mean(dim);
	for (auto &x: X)
		add_self(data_mean, x);
	for (auto &v: data_mean)
		v /= X.size();
	for (auto &x: X) {
		auto v = x;
		sub_self(v, data_mean);
		for (auto &u: v)
			u = u * u;
		add_self(initial_sigma, v);
	}
	mult_self(initial_sigma, 1.0 / (X.size() - 1));
	for (auto &v: initial_sigma)
		v = sqrt(v);

	gmm->gaussians.resize(gmm->nr_mixtures);
	for (auto &g: gmm->gaussians)
		g = new Gaussian(dim, gmm->covariance_type);

	if (init_with_kmeans) {
		// kmeans
		std::vector<std::vector<real_t>> centroids;
		Dataset X_sparse;
		KMeansIISolver kmeans(concurrency);
		Dense2Sparse(X, X_sparse);
		kmeans.cluster(X_sparse, centroids, gmm->nr_mixtures);
		assert((int)centroids.size() == gmm->nr_mixtures);

		for (int i = 0; i < gmm->nr_mixtures; i ++) {
			auto &g = gmm->gaussians[i];
			assert(g->mean.size() == centroids[i].size());
			g->mean = centroids[i];
		}
	} else {
		for (int i = 0; i < gmm->nr_mixtures; i ++) {
			auto &g = gmm->gaussians[i];
			g->mean = X[random.rand_int(X.size())];
		}
	}

	for (auto &g: gmm->gaussians) {
		g->sigma = initial_sigma;
	}

	gmm->weights.resize(gmm->nr_mixtures);
	for (auto &w: gmm->weights) {
		w = 1.0 / gmm->nr_mixtures;
	}
	gmm->normalize_weights();
}

void GMM::normalize_weights() {
	real_t w_sum = 0;
	for (auto &w: weights)
		w_sum += w;
	for (auto &w: weights)
		w /= w_sum;
}

void GMMTrainerBaseline::clear_gaussians() {
	for (auto &g: gmm->gaussians)
		delete g;
	vector<Gaussian *>().swap(gmm->gaussians);
}

static void gassian_set_zero(Gaussian *gaussian) {
	for (auto &m: gaussian->mean)
		m = 0;
	for (auto &s: gaussian->sigma)
		s = 0;
	for (auto &row: gaussian->covariance)
		for (auto &v: row)
			v = 0;

}

void GMMTrainerBaseline::update_weights(std::vector<std::vector<real_t>> &) {
	for (int k = 0; k < gmm->nr_mixtures; k ++) {
//      gmm->weights[k] = (N_k[k] + EPS * 10) / n + EPS;
		gmm->weights[k] = N_k[k] / n;
	}
	gmm->normalize_weights();
}

void GMMTrainerBaseline::update_means(std::vector<std::vector<real_t>> &X) {
	Threadpool pool(concurrency);
	for (int k = 0; k < gmm->nr_mixtures; k ++) {
		auto task = [&](int k) {
			vector<real_t> tmp(dim);
			auto &gaussian = gmm->gaussians[k];
			for (auto &m: gaussian->mean)
				m = 0;
			for (int i = 0; i < n; i ++) {
				mult(X[i], prob_of_y_given_x[k][i], tmp);
				add_self(gaussian->mean, tmp);
			}
			mult_self(gaussian->mean, 1.0 / N_k[k]);
		};
		pool.enqueue(bind(task, k), 1);
	}
}


void GMMTrainerBaseline::update_variance(std::vector<std::vector<real_t>> &X) {
	real_t min_sigma = sqrt(min_covar);
	Threadpool pool(concurrency);
	for (int k = 0; k < gmm->nr_mixtures; k ++) {
		auto task = [&](int k) {
			vector<real_t> tmp(dim);
			auto &gaussian = gmm->gaussians[k];
			for (auto &v: gaussian->sigma) v = 0;
			for (int i = 0; i < n; i ++) {
				sub(X[i], gaussian->mean, tmp);
				for (auto &t: tmp) t = t * t;
				mult_self(tmp, prob_of_y_given_x[k][i]);
				add_self(gaussian->sigma, tmp);
			}
			mult_self(gaussian->sigma, 1.0 / N_k[k]);
			for (auto &s: gaussian->sigma) {
				s = sqrt(s);
				s = max(min_sigma, s);
			}
		};
		pool.enqueue(bind(task, k), 1);
	}
}

void GMMTrainerBaseline::iteration(std::vector<std::vector<real_t>> &X) {
	n = (int)X.size();

	bool enable_guarded_timer = verbosity >= 2;
	{
		GuardedTimer timer("calculate probability of y given x", enable_guarded_timer);

		double **buffers = NULL;
		int batch_size = (int)ceil(gmm->nr_mixtures / (double)concurrency);
		int nr_batch = gmm->nr_mixtures; //(int)ceil(gmm->nr_mixtures / (double)batch_size)
		buffers = new double *[nr_batch];
		for (int i = 0; i < nr_batch; i ++)
			buffers[i] = new double[gmm->gaussians[0]->fast_gaussian_dim];

		{
			Threadpool pool(concurrency);
			for (int k = 0, bid = 0; k < gmm->nr_mixtures; k += batch_size, bid ++) {
				assert(bid < nr_batch);
				auto task = [&](int begin, int end, int bid){
					double *buffer = NULL;
					buffer = buffers[bid];
					for (int k = begin; k < end; k ++) {
						for (int i = 0; i < n; i ++) {
							real_t pdf_x;
							pdf_x = gmm->gaussians[k]->probability_of_fast_exp(X[i], buffer);
							prob_of_y_given_x[k][i] = gmm->weights[k] * pdf_x;
						}
					}
				};
				pool.enqueue(bind(task, k, min(k + batch_size, gmm->nr_mixtures), bid), 1);
			}
		}
		if (buffers) {
			for (int i = 0; i < nr_batch; i ++)
				delete [] buffers[i];
			delete [] buffers;
		}
	}

	{
		GuardedTimer timer("normalize probability", enable_guarded_timer);
		{
			Threadpool pool(concurrency);
			static const double MIN_PROB_SUM = 1e-15;
			for (int i = 0; i < n; i ++) {
				auto task = [&](int i) {
					real_t &sum = prob_sum[i] = 0;
					for (int k = 0; k < gmm->nr_mixtures; k ++)
						sum += prob_of_y_given_x[k][i];
					if (!(sum > 0)) {
						sum = MIN_PROB_SUM;
					}
					assert(sum > 0);
				};
				pool.enqueue(bind(task, i), 1);
			}
		}
		for (int k = 0; k < gmm->nr_mixtures; k ++)
			for (int i = 0; i < n; i ++)
				prob_of_y_given_x[k][i] /= prob_sum[i];
	}

	{
		const double min_n_k = 1e-6;
		GuardedTimer timer("calculate N_k", enable_guarded_timer);
		for (int k = 0; k < gmm->nr_mixtures; k ++) {
			N_k[k] = 0;
			for (int i = 0; i < n; i ++)
				N_k[k] += prob_of_y_given_x[k][i];
			if (N_k[k] == 0)
				N_k[k] = min_n_k;
			if (!(N_k[k] > 0))
				printf("N_k[k] = %f\n", N_k[k]);
			assert(N_k[k] > 0);
		}
	}

	{
		GuardedTimer timer("calculate weights", enable_guarded_timer);
		update_weights(X);

	}
	{
		GuardedTimer timer("update means", enable_guarded_timer);
		update_means(X);

	}

	{
		GuardedTimer timer("update variance", enable_guarded_timer);
		update_variance(X);
	}
}

static void threaded_log_probability_of(GMM *gmm, std::vector<std::vector<real_t>> &X, std::vector<real_t> &prob_buffer, int concurrency) {
	int n = (int)X.size();
	prob_buffer.resize(n);
	int batch_size = (int)ceil(n / (real_t)concurrency);

	int nr_batch = (int)ceil(n / (double)batch_size) ;
	double **buffers = new double *[nr_batch];
	for (int i = 0; i < nr_batch; i ++)
		buffers[i] = new double[gmm->gaussians[0]->fast_gaussian_dim];

	{
		Threadpool pool(concurrency);

		for (int i = 0, id = 0; i < n; i += batch_size, id ++) {
			auto task = [&](int begin, int end, double *buffer){
				for (int j = begin; j < end; j ++) {
					prob_buffer[j] = gmm->log_probability_of_fast_exp(X[j], buffer);
				}
			};
			pool.enqueue(bind(task, i, min(i + batch_size, n), buffers[id]), 1);
		}

	}

	for (int i = 0; i < nr_batch; i ++)
		delete [] buffers[i];
	delete [] buffers;
}

static real_t threaded_log_probability_of(GMM *gmm, std::vector<std::vector<real_t>> &X, int concurrency) {
	std::vector<real_t> prob_buffer;
	threaded_log_probability_of(gmm, X, prob_buffer, concurrency);
	real_t prob = 0;
	for (auto &p: prob_buffer)
		prob += p;
	return prob;
}

real_t GMM::log_probability_of_fast_exp_threaded(std::vector<std::vector<real_t>> &X, int concurrency) {
	return threaded_log_probability_of(this, X, concurrency);
}

void GMM::log_probability_of_fast_exp_threaded(
		std::vector<std::vector<real_t>> &X, std::vector<real_t> &prob_out, int concurrency) {
	threaded_log_probability_of(this, X, prob_out, concurrency);
}


void GMMTrainerBaseline::train(GMM *gmm, std::vector<std::vector<real_t>> &X) {
	if (X.size() == 0) {
		const char *msg = "X.size() == 0";
		printf("%s\n", msg);
		throw msg;
	}

	this->gmm = gmm;
	gmm->dim = dim = X[0].size();

	clear_gaussians();
	init_gaussians(X);

	prob_of_y_given_x.resize(gmm->nr_mixtures);
	for (auto &v: prob_of_y_given_x)
		v.resize(X.size());

	prob_sum.resize(X.size());

	N_k.resize(gmm->nr_mixtures);


#define PAUSE() \
	do { \
		ofstream out("gmm-test.model"); \
		gmm->dump(out); \
		gmm->dump(cout); \
		printf("press a key to continue ...\n"); \
		getchar(); \
	} while (0)

	real_t last_ll = -numeric_limits<real_t>::max();
	for (int i = 0; i < nr_iter; i ++) {
		GuardedTimer iter_time("iteration total time", verbosity >= 2);
		Timer timer;
		timer.start();
		iteration(X);
		if (verbosity >= 1) {
			printf("iteration time: %.3lfs\n", timer.stop() / 1000.0); fflush(stdout);
		}

		if (i % 2 == 0)
			continue;
		if (true || gmm->nr_mixtures * (long long)X.size() >= 43520000) {
			string dump_file = "gmm-training-intermediate-dump.model";
			printf("dumping model to %s ...\n", dump_file.c_str());
			ofstream of(dump_file);
			gmm->dump(of);
			printf("model dumped to %s ...\n", dump_file.c_str());
		}
		// monitor average log likelihood
		timer.start();
		real_t ll;
		//        ll = gmm->log_probability_of(X);
		ll = threaded_log_probability_of(gmm, X, this->concurrency);
		if (verbosity >= 1) {
			printf("log_probability_of time: %.3lfs\n", timer.stop() / 1000.0); fflush(stdout);
		}
		if (verbosity >= 1) {
			printf("iter %d: ll %lf\n", i, ll); fflush(stdout);
		}

		real_t ll_diff = ll - last_ll;
		if (fabs(ll_diff) / fabs(ll) < threshold && ll_diff < threshold) {
			if (verbosity >= 1) {
				printf("too small log likelihood increment, abort iteration.\n");
				fflush(stdout);
			}
			break;
		}
		last_ll = ll;
	}
}

void GMM::dump(ostream &out) {
	out << nr_mixtures << endl;
	for (auto &w: weights)
		out << w << ' ';
	out << endl;
	for (auto &g: gaussians)
		g->dump(out);
}

void GMM::load(istream &in) {
	dim = 0; // of no importance, since it will be overwritten
	in >> nr_mixtures;
	weights.resize(nr_mixtures);
	for (auto &w: weights)
		in >> w;
	gaussians.resize(nr_mixtures);
	int cnt = 0;
	try{
		for (auto &g: gaussians) {
			g = new Gaussian(dim, COVTYPE_DIAGONAL);
			g->load(in);
			dim = g->dim;
		}
	} catch (const char *msg) {
		printf("error loading gmm: %s\n", msg);
		throw msg;
	}
}

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

