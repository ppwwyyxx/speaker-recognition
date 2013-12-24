/*
 * $File: main.cc
 * $Date: Tue Dec 24 18:42:29 2013 +0800
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

static void svm2vec(const std::vector<std::pair<int, real_t>> &input,
		std::vector<real_t> &output, int dim) {
	output = std::vector<real_t>(dim, 0);
	for (auto &item: input) {
		if (item.first > dim) // ignore out bounded values
			continue;
		output[item.first] = item.second;
	}
}

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

void Dataset2DenseDataset(Dataset &X0, DenseDataset &X1) {
	int n, m;
	get_data_metric(X0, n, m);

	X1.resize(X0.size());
	for (size_t i = 0; i < X0.size(); i ++)
		svm2vec(X0[i], X1[i], m);
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

void fill_gaussian(DenseDataset &X, Gaussian *gaussian, int nr_point) {
	for (int i = 0; i < nr_point; i ++)
		X.push_back(gaussian->sample());
}

static vector<real_t> random_vector(int dim, real_t range, Random &random) {
	vector<real_t> vec(dim);
	for (auto &v: vec) v = random.rand_real() * range;
	return vec;
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

void gen_gaussian_mixture(DenseDataset &X, int nr_point_per_gaussian = 1000) {
	int nr_gaussian = 3;
	Gaussian g0(2);
	g0.mean = {0, 0};
	g0.sigma = {0.1, 0.1};

	Gaussian g1(2);
	g1.mean = {1, 1};
	g1.sigma = {0.1, 0.1};

	Gaussian g2(2);
	g2.mean = {2, 1};
	g2.sigma = {0.2, 0.2};

	fill_gaussian(X, &g0, nr_point_per_gaussian);
	fill_gaussian(X, &g1, nr_point_per_gaussian);
	fill_gaussian(X, &g2, nr_point_per_gaussian);
}

void simple_test(std::vector<std::vector<real_t>> &X) {
	int concurrency = 4;
	int nr_iter = 10;
	int nr_mixture = 1024;
	GMMTrainerBaseline trainer(nr_iter, 1e-3, 0.01, 0, concurrency);
	trainer.verbosity = 2;
	GMM gmm(nr_mixture, COVTYPE_DIAGONAL, &trainer);
	printf("start training ...\n"); fflush(stdout);
	gmm.fit(X);
}

int main(int argc, char *argv[]) {
	srand(42); // Answer to The Ultimate Question of Life, the Universe, and Everything


	DenseDataset X;
	read_dense_dataset(X, "test.data");

	simple_test(X);
	return 0;

//    gen_gaussian_mixture(X, 256);
//    gen_high_dim_gaussian_mixture(X, 13, 10, 68000);

//    write_dense_dataset(X, "test.data");

	int nr_mixture = 256;
	int nr_iter = 10;

	for (int concurrency = 1; concurrency < 16; concurrency ++) {
		for (int nr_instance = 1000; nr_instance <= 512000 && nr_instance < (int)X.size(); nr_instance *= 2) {
			GMMTrainerBaseline trainer(nr_iter, 1e-3, 0.01, 0, concurrency);
			GMM gmm(nr_mixture, COVTYPE_DIAGONAL, &trainer);
			printf("start training ...\n"); fflush(stdout);
			DenseDataset X_train(nr_instance);
			for (int i = 0; i < nr_instance; i ++)
				X_train[i] = X[i];
			clock_t start = clock();
			gmm.fit(X_train);
			double duration = (clock() - start) / (double)CLOCKS_PER_SEC / concurrency;
			printf("result pygmm %d %f %d\n", nr_instance, duration, concurrency);
		}

	}
//    ofstream fout("gmm-test.model");
//    gmm.dump(fout);

	return 0;
}

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

