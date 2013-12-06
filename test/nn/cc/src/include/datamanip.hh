/*
 * $File: datamanip.hh
 * $Date: Thu Dec 05 00:08:15 2013 +0000
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#pragma once

#include "dataset.hh"

#include <cstdio>
#include <cassert>
#include "random.hh"

// integer label
void read_svm_data(const char *fpath, Dataset &dataset, Labels &labels);
void read_svm_data(const char *fpath, Dataset &dataset, RealLabels &labels);
void print_data(FILE *fout, const Dataset &dataset);
void print_data(FILE *fout, const Dataset &dataset, const Labels &labels);
void print_data(FILE *fout, const Dataset &dataset, const RealLabels &labels);
void print_labels(FILE *fout, const Labels &labels);
void print_labels(FILE *fout, const RealLabels &labels);
void print_instance(FILE *fout, const Instance &instance);

void get_data_metric(const Dataset &x, int &n, int &m);
void get_data_metric(const std::vector<std::vector<real_t> >&x, int &n, int &m);

template<typename Dataset>
void get_refdata_metric(const Dataset &x, int &n, int &m) {
	n = x.size();
	m = 0;
	for (size_t i = 0; i < x.size(); i ++) {
		auto &xi = *x[i];
		for (size_t j = 0; j < xi.size(); j ++)
			if (xi[j].first + 1 > m)
				m = xi[j].first + 1;
	}
}

// n_choose == -1 means the same number as dataset.size()
template<typename Dataset_t, typename Labels_t>
void bootstrap_samples(const Dataset_t &dataset, const Labels_t &labels,
		Dataset_t &subdataset, Labels_t &sublabels, int n_choose = -1) {
	Random random;
	assert(dataset.size() == labels.size());
	subdataset.resize(0); sublabels.resize(0);
	if (n_choose == -1)
		n_choose = dataset.size();
	for (int i = 0; i < n_choose; i ++) {
		int index = random.rand_int(dataset.size());
		subdataset.push_back(dataset[index]);
		sublabels.push_back(labels[index]);
	}
}

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

