/*
 * $File: main.cc
 * $Date: Sat Dec 07 15:42:24 2013 +0000
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
#include <cstdlib>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>

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

void fill_circle_data(DenseDataset &X, double x0 = 0.0, double y0 = 0.0, double r = 1.0) {
	Random random;
	int nr_point = 1000;
	for (int i = 0; i < nr_point; i ++) {
		real_t angle = random.rand_real() * 2 * M_PI;
		real_t x = cos(angle), y = sin(angle);
		real_t dx = random.rand_normal() * x * 0.1;
		real_t dy = random.rand_normal() * y * 0.1;
		x += dx;
		y += dy;

		x = x * r + x0, y = y * r + y0;

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

void fill_line(DenseDataset &X, double x0, double y0, double x1, double y1, int nr_point = 1000) {
	Random random;
	for (int i = 0; i < nr_point; i ++) {
		double t = random.rand_real();
		double x = t * (x1 - x0),
			   y = t * (y1 - y0);

		double l = sqrt(x * x + y * y);
		double dx = x / l, dy = y / l;
		dx = -dy * random.rand_normal() * 0.01;
		dy = dx * random.rand_normal() * 0.01;
		x += dx, y += dy;
		x += x0, y += y0;
		X.push_back({x, y});
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
		if (end == len || line[end] == '\n')
			break;
		begin = end + 1;
		end = begin;
	}
	return x;
}

void read_dataset(DenseDataset &X, const char *fname) {
	ifstream fin(fname);
	string line;
	while (getline(fin, line)) {
		X.push_back(string_to_double_vector(line));
	}
}
void write_dataset(DenseDataset &X, const char *fname) {
	ofstream fout(fname);
	for (auto &x: X) {
		for (auto &v: x)
			fout << v << ' ';
		fout << endl;
	}
}

void fill_sin_data(DenseDataset &X) {
	int nr_point = 1000;
	Random random;
	for (int i = 0; i < nr_point; i ++) {
		double x = random.rand_real() * 2 * M_PI;
		double y = sin(x);
		X.push_back({x, y});
	}
}

void test_crbm() {
	DenseDataset X;
	//fill_circle_data(X); print_dataset(X, "circle.data");
//    fill_double_moon(X); print_dataset(X, "double-moon.data");
//    fill_circle_data(X, 0, 0, 1); fill_circle_data(X, 0, 2, 1); print_dataset(X, "double-circle.data");
//    fill_line(X, 0, 0, 1, 1); fill_line(X, 1, 0, 0, 1); print_dataset(X, "cross.data");

//    read_dataset(X, "mfcc-male-0-dim-0-1.data");
	fill_sin_data(X); print_dataset(X, "sin.data");
	string reconstruction_output_file = "crbm-test.reconstruction.sin.data";
	try {
		int nr_h = 15;
		CRBM crbm(nr_h);
		CRBMTrainer trainer;
		trainer.learning_rate = 0.01;
		trainer.nr_epoch_max = 10000000;
		trainer.batch_train_size = 100;
		trainer.CD_k = 1;
		trainer.verbose = true;
		trainer.nr_epoch_report = 100;
		trainer.nr_reconstruction_test = 1500;
		trainer.reconstruction_output_file = reconstruction_output_file;
		trainer.nr_epoch_save = 100;
		trainer.model_file = "crbm-test.model";
		crbm.fit(X, &trainer);
	} catch (const char *errmsg) {
//        printf("%s", errmsg);
		exit(1);
	}
}

vector<string> get_mfcc_file_list(string list_file_name) {
	fstream fin(list_file_name);
	string line;
	vector<string> list;
	while (getline(fin, line))
		list.push_back(line);
	return list;
}

void train_mfcc() {
	double training_utterance_duration = 15;
	double mfcc_frame_shift = 0.01;
	int nr_frames = training_utterance_duration / mfcc_frame_shift;

	auto flist = get_mfcc_file_list("mfcc-file-list");
	DenseDataset X;
	for (auto fname: flist) {
		fstream fin(fname);
		string line;
		for (int i = 0; i < nr_frames; i ++) {
			if (!getline(fin, line))
				break;
			X.push_back(string_to_double_vector(line));
		}
	}

	write_dataset(X, "mfcc-train.data");
	printf("#training sample: %d\n", (int)X.size());

	try {
		int nr_h = 50;
		CRBM crbm(nr_h);
		CRBMTrainer trainer;
		trainer.learning_rate = 0.01;
		trainer.nr_epoch_max = 10000000;
		trainer.batch_train_size = 100;
		trainer.CD_k = 6;
		trainer.verbose = true;
		trainer.nr_epoch_report = 1;
		trainer.nr_reconstruction_test = 10000;
		trainer.reconstruction_output_file = "mfcc-crbm.reconstruction.50.data";
		trainer.nr_epoch_save = 2;
		trainer.model_file = "crbm-mfcc.50.model";
		crbm.fit(X, &trainer);
	} catch (const char *errmsg) {
//        printf("%s", errmsg);
		exit(1);
	}
}

void train_single_mfcc() {

	DenseDataset X;
	read_dataset(X, "f_041_03.train.data");

	try {
		int nr_h = 20;
		CRBM crbm(nr_h);
		CRBMTrainer trainer;
		trainer.learning_rate = 0.1;
		trainer.nr_epoch_max = 2000;
		trainer.batch_train_size = X.size();
		trainer.CD_k = 1;
		trainer.verbose = true;
		trainer.nr_epoch_report = 10;
		trainer.nr_reconstruction_test = X.size();
		trainer.reconstruction_output_file = "";
		trainer.nr_epoch_save = 100;
		trainer.model_file = "f_041_03.model";
		crbm.fit(X, &trainer);
	} catch (const char *errmsg) {
//        printf("%s", errmsg);
		exit(1);
	}
}

void test_crbm_mfcc() {
	CRBM rbm;
	vector<string> model_file = {
		"f_001_03.model",
		"f_041_03.model",
	};
	vector<pair<int, string>> test_data = {
		make_pair(0, "f_001_03.test.data"),
		make_pair(1, "f_041_03.test.data"),
	};

	vector<CRBM *> rbms;
	for (auto &mf: model_file) {
		CRBM *rbm = new CRBM();
		rbm->load(mf.c_str());
		rbms.push_back(rbm);
	}
	for (auto &p: test_data) {
		int label = p.first;
		DenseDataset X_test;
		read_dataset(X_test, p.second.c_str());
		vector<real_t> errors;
		real_t min_error = numeric_limits<real_t>::max();
		int pred = -1;
		for (size_t i = 0; i < rbms.size(); i ++) {
			real_t error = rbms[i]->reconstruction_error(X_test) / X_test.size();
			errors.push_back(error);
			if (error < min_error) {
				min_error = error;
				pred = i;
			}
		}
		printf("%d %d ", label, pred);
		for (auto &p: errors)
			printf("%lf ", p);
		printf("\n");
	}

}

CRBM *overall_test_train_crbm(DenseDataset &X, const string &model_file) {

	int nr_h = 25;
	CRBM *crbm_ptr = new CRBM(nr_h);
	CRBM &crbm = *crbm_ptr;
	try {
		CRBMTrainer trainer;
		trainer.learning_rate = 0.1;
		trainer.nr_epoch_max = 2000;
		trainer.batch_train_size = X.size();
		trainer.CD_k = 1;
		trainer.verbose = false;
		trainer.nr_epoch_report = 10;
		trainer.nr_reconstruction_test = X.size();
		trainer.reconstruction_output_file = "";
		trainer.nr_epoch_save = 100;
		trainer.model_file = model_file;
		crbm.fit(X, &trainer);
	} catch (const char *errmsg) {
//        printf("%s", errmsg);
		exit(1);
	}
	return crbm_ptr;
}

void copy_dataset(DenseDataset &X_in, int begin, int end, DenseDataset &X_out) {
	X_out.resize(0);
	for (int i = begin; i < end && i < (int)X_in.size(); i ++)
		X_out.push_back(X_in[i]);
}

string classify(vector<pair<string, CRBM *>> &rbms, DenseDataset &X, vector<real_t> &errors) {
	real_t error = numeric_limits<real_t>::max();
	string pred = "invalid prediction";
	for (auto &rbm: rbms) {
		real_t e = rbm.second->reconstruction_error(X);
		if (e < error) {
			error = e;
			pred = rbm.first;
		}
		errors.push_back(e);
	}
	return pred;
}

void overall_test_get_data(vector<pair<string, DenseDataset>> &training_data,
vector<pair<string, DenseDataset>> &test_data, int nr_data_max = 1000) {
	Random random;

	real_t training_utterance_duration = 15;
	real_t test_utterance_duration = 5;
	real_t mfcc_frame_shift_time = 0.01;
	int nr_training_frames = training_utterance_duration / mfcc_frame_shift_time;
	int nr_test_frames = test_utterance_duration / mfcc_frame_shift_time;
	int nr_tests_per_spkr = 100;

	vector<string> mfcc_flist = get_mfcc_file_list("mfcc-file-list.reading");
	for (auto &fname: mfcc_flist) {
		printf("reading data `%s' ...\n", fname.c_str()); fflush(stdout);
		DenseDataset X, X_train;
		read_dataset(X, fname.c_str());
		string model_file = fname + ".model";
		copy_dataset(X, 0, nr_training_frames, X_train);
		training_data.push_back(make_pair(fname, X_train));
		for (int i = 0; i < nr_tests_per_spkr; i ++){
			DenseDataset X_test;
			int pos = nr_training_frames + random.rand_int(X.size() - nr_training_frames - nr_test_frames);
			if (pos < 0) // some utterances are too short
				continue;
			copy_dataset(X, pos, pos + nr_test_frames, X_test);
			test_data.push_back(make_pair(fname, X_test));
		}
		nr_data_max --;
		if (nr_data_max == 0)
			break;
	}
}

vector<pair<string, CRBM *>> overall_test_train_all_crbm(vector<pair<string, DenseDataset>> &training_data) {
	vector<pair<string, CRBM *>> rbms(training_data.size());
	{
		Threadpool pool(8);
		for (size_t i = 0; i < training_data.size(); i ++) {
			auto &td = training_data[i];
			auto task = [&](pair<string, DenseDataset> &data, int ind){
				string fname = data.first;
				printf("training model `%s' ...\n", fname.c_str()); fflush(stdout);
				string model_file = fname + ".model";
				CRBM *rbm = overall_test_train_crbm(data.second, model_file);
				rbms[ind] = make_pair(fname, rbm);;
			};
//            task(td, i);
			pool.enqueue(bind(task, std::ref(td), i), 1);
		}
	}
	return rbms;
}

void overall_test_do_test(vector<pair<string, CRBM *>> &rbms, vector<pair<string, DenseDataset>> &test_data) {
	int nr_right = 0;
	for (auto &tconf: test_data) {
		string label = tconf.first;
		vector<real_t> errors;
		string pred = classify(rbms, tconf.second, errors);
		printf("%s %s ", label.c_str(), pred.c_str());
		for (size_t i = 0; i < errors.size(); i ++) {
			real_t e = errors[i];
			if (rbms[i].first == label) printf("<");
			if (rbms[i].first == pred) printf("[");
			printf("%lf", e);
			if (rbms[i].first == pred) printf("]");
			if (rbms[i].first == label) printf(">");
			printf(" ");
		}
		bool right = (label == pred);
		nr_right += right;
		printf("%s" , right ? "" : "wrong");
		printf("\n");
		fflush(stdout);
	}

	printf("%d/%d %lf\n", nr_right, (int)test_data.size(),
			nr_right / (double)test_data.size());
}

void overall_test() {
	vector<pair<string, DenseDataset>> test_data;
	vector<pair<string, DenseDataset>> training_data;
	overall_test_get_data(training_data, test_data, 20);
	auto rbms = overall_test_train_all_crbm(training_data);
	overall_test_do_test(rbms, test_data);
}

int main(int argc, char *argv[]) {
//    test_rbm();
//    test_crbm();

//    train_mfcc();
//    train_single_mfcc();
//    test_crbm_mfcc();
	overall_test();
	return 0;
}

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

