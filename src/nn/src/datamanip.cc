/*
 * $File: datamanip.cc
 * $Date: Thu Dec 05 00:08:18 2013 +0000
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#include "datamanip.hh"

#include "common.hh"

#include <cstdio>
#include <cstdlib>

using namespace std;

static const int BUF_SIZE = 65536;

void read_svm_data(const char *fpath, Dataset &dataset, Labels &labels) {
	FILE *fin = fopen(fpath, "r");
	dataset.resize(0);
	labels.resize(0);

	char *buf = new char[BUF_SIZE];
	while (fgets(buf, BUF_SIZE, fin) == buf) {
		Instance x;
		char *ptr;
		for (ptr = buf; *ptr; ptr ++) {
			if (*ptr == ' ' || *ptr == '\n' || *ptr == '\r') {
				*ptr = 0;
				int label = atoi(buf);
				/*
				if (label != 0 && label != 1) {
					printf("!!!!%d\n", label);
				}
				*/
				labels.push_back(label);
				*ptr = ' ';
				break;
			}
		}
		char *last = ptr;
		int ind = -1;
		double val;
		for (; ; ptr ++) {
			if (*ptr == ' ' || *ptr == '\n' || *ptr == '\r' || *ptr == 0) {
				if (ind != -1) {
					char orig = *ptr;
					*ptr = 0;
					val = atof(last);
					x.push_back(make_pair(ind, val));
					*ptr = orig;
					ind = -1;
				}
				last = ptr + 1;
			} else if (*ptr == ':') {
				*ptr = 0;
				ind = atoi(last);
				last = ptr + 1;
				*ptr = ':';
			}
			if (*ptr == 0)
				break;
		}
		dataset.push_back(x);
	}
	delete [] buf;
	fclose(fin);
}

void read_svm_data(const char *fpath, Dataset &dataset, RealLabels &labels) {
	FILE *fin = fopen(fpath, "r");
	dataset.resize(0);
	labels.resize(0);

	char *buf = new char[BUF_SIZE];
	while (fgets(buf, BUF_SIZE, fin) == buf) {
		Instance x;
		char *ptr;
		for (ptr = buf; *ptr; ptr ++) {
			if (*ptr == ' ' || *ptr == '\n' || *ptr == '\r') {
				*ptr = 0;
				real_t label = atof(buf);
				/*
				if (label != 0 && label != 1) {
					printf("!!!!%d\n", label);
				}
				*/
				labels.push_back(label);
				*ptr = ' ';
				break;
			}
		}
		char *last = ptr;
		int ind = -1;
		double val;
		for (; ; ptr ++) {
			if (*ptr == ' ' || *ptr == '\n' || *ptr == '\r' || *ptr == 0) {
				if (ind != -1) {
					char orig = *ptr;
					*ptr = 0;
					val = atof(last);
					x.push_back(make_pair(ind, val));
					*ptr = orig;
					ind = -1;
				}
				last = ptr + 1;
			} else if (*ptr == ':') {
				*ptr = 0;
				ind = atoi(last);
				last = ptr + 1;
				*ptr = ':';
			}
			if (*ptr == 0)
				break;
		}
		dataset.push_back(x);
	}
	delete [] buf;
	fclose(fin);
}

void print_instance(FILE *fout, const Instance &instance) {
	for (auto item: instance)
		fprintf(fout, "%d:%lf ", item.first, item.second);
}

void print_data(FILE *fout, const Dataset &dataset) {
	for (auto instance: dataset) {
		print_instance(fout, instance);
		fprintf(fout, "\n");
	}
}

void print_data(FILE *fout, const Dataset &dataset, const Labels &labels) {
	assert(dataset.size() == labels.size());
	For(i, dataset.size()) {
		fprintf(fout, "%d ", labels[i]);
		print_instance(fout, dataset[i]);
		fprintf(fout, "\n");
	}
}

void print_data(FILE *fout, const Dataset &dataset, const RealLabels &labels) {
	assert(dataset.size() == labels.size());
	For(i, dataset.size()) {
		fprintf(fout, "%f ", labels[i]);
		print_instance(fout, dataset[i]);
		fprintf(fout, "\n");
	}
}

void print_labels(FILE *fout, const Labels &labels) {
	for (auto i: labels)
		fprintf(fout, "%d\n", i);
}

void print_labels(FILE *fout, const RealLabels &labels) {
	for (auto i: labels)
		fprintf(fout, "%f\n", i);
}

void get_data_metric(const Dataset &x, int &n, int &m) {
	n = x.size();
	m = 0;
	for (size_t i = 0; i < x.size(); i ++) {
		const Instance &xi = x[i];
		for (size_t j = 0; j < xi.size(); j ++)
			if (xi[j].first + 1 > m)
				m = xi[j].first + 1;
	}
}

void get_data_metric(const std::vector<std::vector<real_t> >&x, int &n, int &m) {
	if (x.size() == 0) {
		n = m = -1;
		return;
	}
	n = x.size();
	m = x[0].size();
	for (size_t i = 0; i < x.size(); i ++)
		if ((int)x[i].size() != m) {
			n = m = -1;
			break;
		}
}





/**
 * vim: syntax=cpp11 foldmethod=marker
 */

