/*
 * $File: kmeans.cc
 * $Date: Sun Dec 15 20:22:37 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#include "kmeans.hh"
#include "Threadpool/Threadpool.hpp"

#include <cassert>
#include <algorithm>
#include <functional>
#include <thread>
#include <mutex>
#include <queue>

using ThreadLib::Threadpool;

using namespace std;

static void analyse_dataset(const Dataset &dataset, int &n, int &m) {
	n = dataset.size();
	m = 0;
	for (auto point: dataset)
		for (auto dim: point)
			if (dim.first > m)
				m = dim.first;
	m ++;
}

KMeansSolver::KMeansSolver(int concurrency) {
	this->concurrency = concurrency;
}

static void Instance2Vector(const Instance &instance, Vector &ret, int dim) {
	ret.resize(dim);
	for (auto &item: instance)
		ret[item.first] = item.second;
}

real_t KMeansSolver::cluster(const Dataset &dataset, std::vector<Vector> &centroids, int K) {
	// trivial initialization
	int n, m;
	analyse_dataset(dataset, n, m);
	centroids.resize(K);
	for (auto &p: centroids)
		p.resize(m);

	// randomly choose K points with equal probability
	vector<int> perm(min(n, K));
	for (int i = 0; i < n && i < K; i ++) perm[i] = i;
	random_shuffle(perm.begin(), perm.end());
	for (int i = 0; i < n && i < K; i ++)
		Instance2Vector(dataset[i], centroids[i], m);

	return Lloyds_iteration(dataset, centroids);
}

namespace KMeansSolverImpl {

	real_t distancesqr(const Instance &inst, const Vector &point) {
		real_t dist = 0;
		for (auto &item: inst) {
			real_t delta = item.second - point[item.first];
			dist += delta * delta;
		}
		return dist;
	}

	// return distsqr sum
	real_t calc_belonging(const Dataset &dataset,
			const vector<Vector> &centroids,
			vector<Vector> &new_centroids, // this parameter is exclusive
			vector<int> &cluster_size, // this parameter is exclusive
			int begin, int end) {

		// initialize
		for (auto &i: cluster_size) i = 0;
		for (auto &vec: new_centroids)
			for (auto &point: vec)
				point = 0;


		real_t distsqr_sum = 0;

		for (int i = begin; i < end; i ++) {
			const Instance &inst = dataset[i];
			real_t min_distsqr = numeric_limits<real_t>::max();
			int min_dist_id = -1;
			for (size_t j = 0; j < centroids.size(); j ++) { // K
				auto &vec = centroids[j];
				real_t distsqr = distancesqr(inst, vec);
				if (distsqr < min_distsqr) {
					min_distsqr = distsqr;
					min_dist_id = j;
				}
			}

			cluster_size[min_dist_id] += 1;
			auto &centroid = new_centroids[min_dist_id];
			for (auto &item: inst)
				centroid[item.first] += item.second;
			distsqr_sum += min_distsqr;
		}
		return distsqr_sum;
	}

	real_t calc_belonging_weighted(const Dataset &dataset,
			const vector<Vector> &centroids,
			const vector<real_t> &weight,
			vector<Vector> &new_centroids, // this parameter is exclusive
			vector<real_t> &cluster_size, // this parameter is exclusive
			int begin, int end) {

		// initialize
		for (auto &i: cluster_size) i = 0;
		for (auto &vec: new_centroids)
			for (auto &point: vec)
				point = 0;


		real_t distsqr_sum = 0;

		for (int i = begin; i < end; i ++) {
			const Instance &inst = dataset[i];
			real_t min_distsqr = numeric_limits<real_t>::max();
			int min_dist_id = -1;
			for (size_t j = 0; j < centroids.size(); j ++) { // K
				auto &vec = centroids[j];
				real_t distsqr = distancesqr(inst, vec) * weight[i];
				if (distsqr < min_distsqr) {
					min_distsqr = distsqr;
					min_dist_id = j;
				}
			}

			cluster_size[min_dist_id] += weight[i];
			auto &centroid = new_centroids[min_dist_id];
			for (auto &item: inst)
				centroid[item.first] += item.second * weight[i];
			distsqr_sum += min_distsqr;
		}
		return distsqr_sum;
	}
}

using namespace KMeansSolverImpl;

real_t KMeansSolver::Lloyds_iteration(
		const Dataset &dataset, std::vector<Vector> &centroids) {

	printf("entering Lloyds_iteration\n"); fflush(stdout);

	Threadpool p(concurrency);
	printf("concu..........################: %d\n", concurrency);
	int n = dataset.size();
	int m = centroids[0].size();
	int K = centroids.size();

	vector<Vector> centroids_best;
	real_t distsqr_best = numeric_limits<real_t>::max();

	vector<int> belong(n);

	int block_size = ceil((double)n / concurrency);

	vector<vector<int>> cluster_size_buf(concurrency, vector<int>(K));
	vector<vector<Vector>> centroids_buf(concurrency, vector<Vector>(K, Vector(m)));

	real_t last_distsqr_sum = numeric_limits<real_t>::max();
	const int n_iter_max = 200;
	for (int iter_id = 0; iter_id < n_iter_max; iter_id ++) {
		vector<future<real_t>> results;
		// Calculate belonging
		for (int i = 0, c = 0; i < n; i += block_size, c ++) {
			assert(c < concurrency);
			results.emplace_back(p.enqueue(bind(calc_belonging,
							std::cref(dataset),
							std::cref(centroids), std::ref(centroids_buf[c]),
							std::ref(cluster_size_buf[c]),
							i, min(n, i + block_size)), 0));
		}

		real_t distsqr_sum = 0;
		for (auto &rst: results)
			distsqr_sum += rst.get();

		if (distsqr_sum < distsqr_best) {
			distsqr_best = distsqr_sum;
			centroids_best = centroids;
		}
		printf("iteration %3d: %f\n", iter_id, distsqr_sum); fflush(stdout);
		if (fabs(last_distsqr_sum - distsqr_sum) < 1e-6) { // || fabs(distsqr_sum - last_distsqr_sum) / last_distsqr_sum < 1e-6)
			printf("distance squared sum not changing, converged.\n");
			break;
		}

		const real_t terminate_cost_factor = 1.5;
		if (distsqr_sum > distsqr_best * terminate_cost_factor) {
			printf("distance square sum %f is worse than best found by a factor of %f. ternimating.\n", distsqr_sum, terminate_cost_factor);
			break;
		}


		// aggregate cluster_size
		vector<real_t> cluster_size(K);
		for (auto &cs: cluster_size_buf) {
			assert(cs.size() == (size_t)K);
			for (int i = 0; i < K; i ++)
				cluster_size[i] += cs[i];
		}

		for (auto &point: centroids)
			for (auto &coord: point)
				coord = 0;

		// aggregate centroids
		for (auto ctrds: centroids_buf) {
			assert(ctrds.size() == (size_t)K);
			assert(centroids.size() == (size_t)K);
			for (int i = 0; i < K; i ++) { // id of centroid
				assert(centroids[i].size() == (size_t)m);
				assert(ctrds[i].size() == (size_t)m);
				for (int j = 0; j < m; j ++) // coord index
					centroids[i][j] += ctrds[i][j];
			}
		}

		// compute new centroids
		for (int i = 0; i < K; i ++)
			for (int j = 0; j < m; j ++)
				centroids[i][j] /= (real_t)cluster_size[i];

		last_distsqr_sum = distsqr_sum;

		if (iter_id == n_iter_max - 1) {
			printf("max number of iterations(%d) reached, quit iteration\n", n_iter_max); fflush(stdout);
		}
	}

	centroids = centroids_best;
	printf("minimum distance squared sum(Lloyds_iteration): %f\n", distsqr_best); fflush(stdout);

	return distsqr_best;
}


real_t KMeansSolver::Lloyds_iteration_weighted(
		const Dataset &dataset, const std::vector<real_t> &weight,
		std::vector<Vector> &centroids) {

	printf("entering Lloyds_iteration_weighted\n"); fflush(stdout);

	Threadpool p(concurrency);
	int n = dataset.size();


	int m = centroids[0].size();
	int K = centroids.size();

	vector<Vector> centroids_best;
	real_t distsqr_best = numeric_limits<real_t>::max();

	vector<int> belong(n);

	int block_size = ceil((double)n / concurrency);

	vector<vector<real_t>> cluster_size_buf(concurrency, vector<real_t>(K));
	vector<vector<Vector>> centroids_buf(concurrency, vector<Vector>(K, Vector(m)));

	const int n_iter_max = 200;

	real_t last_distsqr_sum = numeric_limits<real_t>::max();
	for (int iter_id = 0; iter_id < n_iter_max; iter_id ++) {
		vector<future<real_t>> results;
		// Calculate belonging
		for (int i = 0, c = 0; i < n; i += block_size, c ++) {
			assert(c < concurrency);
			results.emplace_back(p.enqueue(bind(calc_belonging_weighted,
							std::cref(dataset),
							std::cref(centroids),
							std::cref(weight),
							std::ref(centroids_buf[c]),
							std::ref(cluster_size_buf[c]),
							i, min(n, i + block_size)), 0));
		}

		real_t distsqr_sum = 0;
		for (auto &rst: results)
			distsqr_sum += rst.get();

		if (distsqr_sum < distsqr_best) {
			distsqr_best = distsqr_sum;
			centroids_best = centroids;
		}

		printf("iteration %3d: %f\n", iter_id, distsqr_sum); fflush(stdout);
		if (fabs(last_distsqr_sum - distsqr_sum) < 1e-6)// || fabs(distsqr_sum - last_distsqr_sum) / last_distsqr_sum < 1e-6)
			break;

		// aggregate cluster_size
		vector<int> cluster_size(K);
		for (auto &cs: cluster_size_buf) {
			assert(cs.size() == (size_t)K);
			for (int i = 0; i < K; i ++)
				cluster_size[i] += cs[i];
		}

		for (auto &point: centroids)
			for (auto &coord: point)
				coord = 0;

		// aggregate centroids
		for (auto ctrds: centroids_buf) {
			assert(ctrds.size() == (size_t)K);
			assert(centroids.size() == (size_t)K);
			for (int i = 0; i < K; i ++) { // id of centroid
				assert(centroids[i].size() == (size_t)m);
				assert(ctrds[i].size() == (size_t)m);
				for (int j = 0; j < m; j ++) // coord index
					centroids[i][j] += ctrds[i][j];
			}
		}

		// compute new centroids
		for (int i = 0; i < K; i ++)
			for (int j = 0; j < m; j ++)
				centroids[i][j] /= cluster_size[i];

		last_distsqr_sum = distsqr_sum;

		if (iter_id == n_iter_max - 1) {
			printf("max number of iterations(%d) reached, quit iteration\n", n_iter_max); fflush(stdout);
		}
	}

	centroids = centroids_best;
	printf("minimum distance squared sum(Lloyds_iteration_weighted): %f\n", distsqr_best); fflush(stdout);
	return distsqr_best;
}


/**
 * vim: syntax=cpp11 foldmethod=marker
 */

