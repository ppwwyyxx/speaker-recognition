/*
 * $File: main.cc
 * $Date: Tue Dec 24 23:34:48 2013 +0000
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#include <cstdio>
#include <fstream>

#include "gmm.hh"

#include "datamanip.hh"
#include "common.hh"

#include "tclap/CmdLine.h"

using namespace std;
using namespace TCLAP;

typedef std::vector<std::vector<real_t>> DenseDataset;

vector<real_t> string_to_double_vector(string line) {
	vector<real_t> x;
	int begin = 0, end = 0;
	int len = line.size();
	while (true) {
		while (end < len && line[end] != ' ' && line[end] != '\n')
			end ++;
		x.push_back(atof(line.substr(begin, end - begin).c_str()));
		if (end == len - 1 || line[end] == '\n' || (end == len - 2 && line[end] == ' ' && line[end] == '\n'))
			break;
		begin = end + 1;
		end = begin;
	}
	return x;
}

struct Args {
	int init_with_kmeans;
	int concurrency;
	real_t threshold;
	int K;
	int iteration;
	real_t min_covar = 1e-3;
	int verbosity;

	string input_file;
	string model_file;
};

Args parse_args(int argc, char *argv[]) {
	Args args;
	try {
		CmdLine cmd("Gaussian Mixture Model (GMM)", ' ', "0.0.1");

		ValueArg<int> arg_init_with_kmeans("f", "kmeans", "init_with_kmeans", false, 1, "NUMBER");
		ValueArg<int> arg_concurrency("w", "concurrency", "number of workers", false, 1, "NUMBER");
		ValueArg<int> arg_K("k", "K", "number of gaussians", true, 10, "NUMBER");
		ValueArg<int> arg_verbosity("v", "V", "verbosity", false, -1, "NUMBER");
		ValueArg<double> arg_min_covar("c", "mincovar", "minimum covariance to avoid overfitting, default 1e-3.", false, 1e-3, "FLOAT");
		ValueArg<double> arg_threshold("t", "threshold", "threshold", false, 0.01, "FLOAT");

		ValueArg<string> arg_input_file("i", "input", "intput file", true, "", "FILE");
		ValueArg<string> arg_model_file("m", "model", "model file", true, "", "FILE");
		ValueArg<int> arg_iteration("r", "iteration", "number of iterations",
				false, 200, "NUMBER");

		cmd.add(arg_concurrency);
		cmd.add(arg_init_with_kmeans);
		cmd.add(arg_K);
		cmd.add(arg_min_covar);
		cmd.add(arg_input_file);
		cmd.add(arg_model_file);
		cmd.add(arg_iteration);
		cmd.add(arg_threshold);
		cmd.add(arg_verbosity);


		cmd.parse(argc, argv);

#define GET_VALUE(name) args.name = arg_##name.getValue();
		GET_VALUE(concurrency);
		GET_VALUE(init_with_kmeans);
		GET_VALUE(K);
		GET_VALUE(min_covar);
		GET_VALUE(input_file);
		GET_VALUE(model_file);
		GET_VALUE(iteration);
		GET_VALUE(verbosity);
		GET_VALUE(threshold);

	} catch (ArgException &e) {
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
	}
	return args;
}

void read_dense_dataset(DenseDataset &X, const char *fname) {
	ifstream fin(fname);
	string line;
	while (getline(fin, line)) {
		X.push_back(string_to_double_vector(line));
	}
}

void write_dense_dataset(DenseDataset &X, const char *fname) {
	ofstream fout(fname);
	for (auto &x: X) {
		for (auto &v: x)
			fout << v << ' ';
		fout << endl;
	}
}

static vector<real_t> random_vector(int dim, real_t range, Random &random) {
	vector<real_t> vec(dim);
	for (auto &v: vec) v = random.rand_real() * range;
	return vec;
}

void fill_gaussian(DenseDataset &X, Gaussian *gaussian, int nr_point) {
	for (int i = 0; i < nr_point; i ++)
		X.push_back(gaussian->sample());
}

void gen_high_dim_gaussian_mixture(DenseDataset &X, int dim, int nr_gaussian, int nr_point_per_gaussian) {
	Random random;
	for (int i = 0; i < nr_gaussian; i ++) {
		Gaussian g(dim);
		g.mean = random_vector(dim, 1, random);
		g.sigma = random_vector(dim, 0.05 + random.rand_real() * 0.1, random);
		fill_gaussian(X, &g, nr_point_per_gaussian);
	}
}


void test() {
	DenseDataset X;
//    read_dense_dataset(X, "test.data");
	gen_high_dim_gaussian_mixture(X, 13, 10, 680);
//    write_dense_dataset(X, "test.data");
	printf("start training");
	GMMTrainerBaseline trainer(10, 1e-3, 0.01, 0, 4, 1);
	GMM ubm(10, COVTYPE_DIAGONAL, &trainer);
	ubm.fit(X);


	DenseDataset().swap(X);
	gen_high_dim_gaussian_mixture(X, 13, 10, 1000);
	GMMUBMTrainerBaseline ubmgmm_trainer(&ubm, 10, 1e-3, 0.01, 4, 1);
	GMM gmm(10, COVTYPE_DIAGONAL, &ubmgmm_trainer);
	printf("training ubm ...\n");
	gmm.fit(X);
}

int main(int argc, char *argv[]) {
	GMM g(10);
	ifstream fin("model/ubm.model");
	g.load(fin);

	srand(42); // Answer to The Ultimate Question of Life, the Universe, and Everything
	test();
	return 0;

	Args args = parse_args(argc, argv);

	DenseDataset X;
	read_dense_dataset(X, args.input_file.c_str());

	GMMTrainerBaseline trainer(
			args.iteration, args.min_covar, args.threshold,
			args.init_with_kmeans, args.concurrency,
			args.verbosity);
	GMM gmm(args.K, COVTYPE_DIAGONAL, &trainer);
	printf("start training ...\n"); fflush(stdout);
	gmm.fit(X);

	ofstream fout(args.model_file);
	gmm.dump(fout);

	return 0;
}

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

