/*
 * $File: dbn.cc
 * $Date: Sun Nov 17 12:57:20 2013 +0000
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#include "dbn.hh"

void DBN::add_rbm(RBM *rbm) {
	rbms.push_back(rbm);
}

void DBN::fit_last_layer(std::vector<std::vector<real_t>> &X) {
	if (rbms.size() > 1) {
		std::vector<std::vector<real_t>> X_filtered(X.size());
		for (size_t i = 0; i < X.size(); i ++) {
			std::vector<real_t> x[2] = {X[i]};
			for (size_t j = 0; j < rbms.size() - 1; j ++) {
				auto rbm = rbms[j];
				rbm->reconstruct(x[j & 1], x[(j + 1) & 1]);
			}
			X_filtered[i] = std::move(x[(rbms.size() - 1) & 1]);
		}
		rbms.back()->fit(X_filtered);
	}
	else { // the first rbm
		rbms.back()->fit(X);
	}
}

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

