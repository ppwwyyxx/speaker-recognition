/*
 * $File: main.cc
 * $Date: Fri Dec 06 23:16:07 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */


#include "datamanip.hh"
#include "neural-net.hh"
#include "rbm.hh"
#include "crbm.hh"
#include "dbn.hh"
#include "Threadpool/Threadpool.hpp"

#include <cstdio>
#include <string>
#include <cmath>
#include <cstdlib>

using namespace std;
using namespace ThreadLib;

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

void Dataset2DenseDataset(Dataset &X0, DenseDataset &X1) {
	int n, m;
	get_data_metric(X0, n, m);

	X1.resize(X0.size());
	for (size_t i = 0; i < X0.size(); i ++)
		svm2vec(X0[i], X1[i], m);
}

void print_vec(std::vector<real_t> &vec) {
	for (auto &v: vec)
		printf("%.2f ", v);
	printf("\n"); fflush(stdout);
}

void print_as_img(std::vector<real_t> &vec, int width, int height, FILE *fout = stdout) {
	assert(vec.size() == (size_t)width * height);
	for (int i = 0; i < height; i ++) {
		for (int j = 0; j < width; j ++) {
			real_t v = round(vec[i * width + j]);
			fprintf(fout, "%c", v > 0 ? '#' : '.');
		}
		fprintf(fout, "\n");
	}
	fflush(fout);
}

struct RBMTrainParam
{
	double learning_rate;
	int nr_epoch;
	int cd_k;
	int batch_size;
	int nr_epoch_to_report;
	int nr_reconstruction_test;
	int nr_epoch_to_save;
	string model_file;
	int verbosity;
	RBMTrainParam() :
		learning_rate(0.1),
		nr_epoch(1000),
		cd_k(1),
		nr_reconstruction_test(100),
		nr_epoch_to_save(20),
		model_file(""),
		verbosity(1)	{
		}
};

double get_reconstruction_error(RBM &rbm, DenseDataset &X, int nr_reconstruction_test) {
	double error = 0;
	vector<real_t> x;
	int nr_test = 0;
	for (int i = 0; i < (int)X.size() && i < nr_reconstruction_test; i ++, nr_test ++) {
		nr_test ++;
		vector<real_t> &v = X[i];
		rbm.reconstruct_light(v, x);
		for (size_t j = 0; j < v.size(); j ++) {
			real_t e = v[j] - x[j];
			error += e * e;
		}
	}
	return error / nr_test;
}

void rbm_train(RBM &rbm, DenseDataset &X, RBMTrainParam &param) {
	rbm.set_CD_k(param.cd_k);
	rbm.set_learning_rate(param.learning_rate);
	int nr_instance = X.size();
	for (int epoch = 0; epoch < param.nr_epoch; epoch ++) {
		if (epoch % param.nr_epoch_to_report == 0)
			printf("epoch %d/%d ...", epoch, param.nr_epoch);

		for (int i = 0; i < nr_instance; i += param.batch_size)
			rbm.fit_batch(X, i, i + param.batch_size);

		if (epoch % param.nr_epoch_to_report == 0)
			printf("reconstruction error: %lf\n",
					get_reconstruction_error(rbm, X, param.nr_reconstruction_test));
		if (epoch % param.nr_epoch_to_save == 0) {
			printf("saving model to `%s'...\n", param.model_file.c_str());
			if (param.model_file.size() != 0) {
				rbm.dump(param.model_file.c_str());
			} else {
				printf("model file not specified, abort saving.\n");
			}
		}
	}
}

void filter_X(DenseDataset &X, DenseDataset &X_filtered,
		RBM &rbm) {
	X_filtered.resize(X.size());
	for (size_t i = 0; i < X.size(); i ++) {
		std::vector<real_t> x(rbm.get_hidden_layer_size()),
			p(rbm.get_hidden_layer_size());
		rbm.sample_hidden_layer(X[i], x, p);
		X_filtered[i] = x;
	}
}

void fill_2D_data(DenseDataset &X) {
	for (int i = 0; i < 100; i ++) {
		X.push_back({0, 1});
		X.push_back({1, 0});
		X.push_back({1, 1});
	}
}
void test_rbm() {

	DenseDataset X;
	fill_2D_data(X);

	int nr_h = 20;
	RBMTrainParam param;
	param.nr_epoch = 100000;
	param.learning_rate = 1.0;
	param.cd_k = 1;
	param.batch_size = 100;
	param.nr_epoch_to_report = 100;
	param.nr_reconstruction_test = 100;
	param.nr_epoch_to_save = 100;
	param.model_file = "rbm-test.model";
	int n, m;
	get_data_metric(X, n, m);
	RBM rbm(m, nr_h);

	rbm_train(rbm, X, param);
}
void print_dataset(DenseDataset &X, const char *fname)  {
	FILE *fout = fopen(fname, "w");
	for (auto &x: X) {
		for (auto &v: x)
			fprintf(fout, "%lf ", v);
		fprintf(fout, "\n");
	}
	fclose(fout);
}

void fill_circle_data(DenseDataset &X) {
	Random random;
	int nr_point = 1000;
	for (int i = 0; i < nr_point; i ++) {
		real_t angle = random.rand_real() * 2 * M_PI;
		real_t x = cos(angle), y = sin(angle);
		real_t dx = random.rand_normal() * x * 0.1;
		real_t dy = random.rand_normal() * y * 0.1;
		x += dx;
		y += dy;

		X.push_back({x, y});
	}
}

void fill_arc(DenseDataset &X, double x0, double y0, double r, double start_angle, double end_angle, int nr_point) {
	while (end_angle < start_angle)
		end_angle += 2 * M_PI;
	Random random;
	for (int i = 0; i < nr_point; i ++) {
		double angle = start_angle + (end_angle - start_angle) * random.rand_real();
		real_t x = cos(angle) * r, y = sin(angle) * r;
		real_t dx = random.rand_normal() * x * 0.1;
		real_t dy = random.rand_normal() * y * 0.1;
		x += dx, y += dy;
		x += x0, y += y0;
		X.push_back({x, y});
	}
}
void fill_double_moon(DenseDataset &X) {
	int nr_point = 1000;
	fill_arc(X, 0, 0, 1, 2 * M_PI / 4, 2 * M_PI / 4 * 3, nr_point);
	fill_arc(X, -0.5, 1, 1, 2 * M_PI / 4 * 3, 2 * M_PI / 4,  nr_point);
}

void test_crbm() {
	DenseDataset X;
	//fill_circle_data(X); print_dataset(X, "circle.data");
	fill_double_moon(X); print_dataset(X, "double-moon.data");

	try {
		int nr_h = 8;
		CRBM crbm(nr_h);
		CRBMTrainer trainer;
		trainer.learning_rate = 0.01;
		trainer.nr_epoch_max = 10000000;
		trainer.batch_train_size = 100;
		trainer.CD_k = 20;
		trainer.verbose = true;
		trainer.nr_epoch_report = 100;
		trainer.nr_reconstruction_test = 1000;
		trainer.reconstruction_output_file = "crbm-test.reconstruction.data";
		trainer.nr_epoch_save = 100;
		trainer.model_file = "crbm-test.model";
		crbm.fit(X, &trainer);
	} catch (const char *errmsg) {
//        printf("%s", errmsg);
		exit(1);
	}
}

int main(int argc, char *argv[]) {
//    test_rbm();
	test_crbm();

	return 0;
}

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

