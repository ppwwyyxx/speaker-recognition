/*
 * $File: util.hh
 * $Date: Tue Dec 24 16:46:50 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#pragma once


#include "type.hh"
#include <cassert>
#include <vector>


using namespace std;

#if 0
static vector<real_t> random_vector(int dim, real_t range, Random &random) {
	vector<real_t> vec(dim);
	for (auto &v: vec) v = random.rand_real() * range;
	return vec;
}
#endif

static void add(const vector<real_t> &a, const vector<real_t> &b, vector<real_t> &c) {
	assert(a.size() == b.size() && b.size() == c.size());
	size_t n = a.size();
	for (size_t i = 0; i < n; i ++)
		c[i] = a[i] + b[i];
}

static void sub(const vector<real_t> &a, const vector<real_t> &b, vector<real_t> &c) {
	assert(a.size() == b.size() && b.size() == c.size());
	size_t n = a.size();
	for (size_t i = 0; i < n; i ++)
		c[i] = a[i] - b[i];
}

#if 0
static void mult(const vector<real_t> &a, const vector<real_t> &b, vector<real_t> &c) {
	assert(a.size() == b.size() && b.size() == c.size());
	size_t n = a.size();
	for (size_t i = 0; i < n; i ++)
		c[i] = a[i] * b[i];
}
#endif

static void mult(const vector<real_t> &a, real_t f, vector<real_t> &b) {
	assert(a.size() == b.size());
	size_t n = a.size();
	for (size_t i = 0; i < n; i ++)
		b[i] = a[i] * f;
}

static void add_self(vector<real_t> &a, const vector<real_t> &b) {
	add(a, b, a);
}

static void sub_self(vector<real_t> &a, const vector<real_t> &b) {
	sub(a, b, a);
}

#if 0
static void mult_self(vector<real_t> &a, const vector<real_t> &b) {
	mult(a, b, a);
}
#endif

static void mult_self(vector<real_t> &a, real_t f) {
	mult(a, f, a);
}

/**
 * vim: syntax=cpp11.doxygen foldmethod=marker
 */

