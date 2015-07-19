/*
 * $File: kmeans++.cc
 * $Date: Wed Dec 11 17:36:56 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#include "kmeans++.hh"
#include "Threadpool/Threadpool.hpp"

#include <cassert>
#include <limits>
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

KMeansppSolver::KMeansppSolver(int concurrency) :
	KMeansSolver(concurrency) {
}

static void Instance2Vector(const Instance &instance, Vector &ret, int dim) {
	ret.resize(dim);
	for (auto &item: instance)
		ret[item.first] = item.second;
}

static void Vector2Instance(const Vector &vec, Instance &instance) {
	instance.resize(0);
	for (size_t i = 0; i < vec.size(); i ++) {
		real_t coord = vec[i];
		if (fabs(coord) < 1e-15)
			continue;
		instance.emplace_back(make_pair(i, coord));
	}
}
namespace KMeansppSolverImpl {
	real_t distancesqr(const Instance &inst, const Vector &point) {
		real_t dist = 0;
		for (auto &item: inst) {
			real_t delta = item.second - point[item.first];
			dist += delta * delta;
		}
		return dist;
	}

	real_t update_distances(
			const Dataset &dataset,
			const Vector &centroid,
			vector<real_t> &distances,
			int begin, int end) {

		real_t distsqr_sum = 0;

		for (int i = begin; i < end; i ++) {
			const Instance &inst = dataset[i];
			real_t &min_distsqr = distances[i];
			real_t distsqr = distancesqr(inst, centroid);
			min_distsqr = min(min_distsqr, distsqr);
			distsqr_sum += min_distsqr;
		}
		return distsqr_sum;
	}

	real_t update_distances_weighted (
			const Dataset &dataset,
			const Vector &centroid,
			const vector<real_t> &weight,
			vector<real_t> &distances,
			int begin, int end) {

		real_t distsqr_sum = 0;

		for (int i = begin; i < end; i ++) {
			const Instance &inst = dataset[i];
			real_t &min_distsqr = distances[i];
			real_t distsqr = distancesqr(inst, centroid) * weight[i];
			min_distsqr = min(min_distsqr, distsqr);
			distsqr_sum += min_distsqr;
		}
		return distsqr_sum;
	}
}

using namespace KMeansppSolverImpl;


real_t KMeansppSolver::cluster(const Dataset &dataset, std::vector<Vector> &centroids, int K) {
	// trivial initialization
	int n, m;
	analyse_dataset(dataset, n, m);
	centroids.resize(K);
	for (auto &p: centroids)
		p.resize(m);

	printf("kmeans++ initializing ...\n");

	// initial cluster with one random point
	Instance2Vector(dataset[random.rand_int() % n], centroids[0], m);

	vector<real_t> distances(n);
	// initialize distances to large numbers
	for (auto &d: distances) d = numeric_limits<real_t>::max();

	Threadpool pool(concurrency);
	int block_size = ceil((double)n / (concurrency));
	for (int k = 1; k < K; k ++) {
		// calculate distances
		vector<future<real_t>> results;
		for (int i = 0, c = 0; i < n; i += block_size, c ++) {
			assert(c < concurrency);
			results.emplace_back(pool.enqueue(bind(update_distances,
							std::cref(dataset),
							std::cref(centroids[k - 1]),
							std::ref(distances),
							i, min(i + block_size, n)), 0));
		}

		real_t distsqr_sum = 0;
		for (auto &rst: results)
			distsqr_sum += rst.get();

#ifdef __DEBUG_CHECK
		real_t distsqr_sum_check = 0;
		for (auto &d: distances) distsqr_sum_check += d;
		assert(distsqr_sum_check == distsqr_sum);
#endif

		real_t random_weight = random.rand_int() / (double)RAND_MAX * distsqr_sum;
		for (int i = 0; i < n; i ++) {
			random_weight -= distances[i];
			if (random_weight <= 0) {
				Instance2Vector(dataset[i], centroids[k], m);
				printf("kmeans++ iteration %3d: %f, #%d has been choosen\n", k, distsqr_sum, i); fflush(stdout);
				break;
			}
		}
	}

	return Lloyds_iteration(dataset, centroids);
}

real_t KMeansppSolver::cluster_weighted(
		const Dataset &dataset, const std::vector<real_t> &weight,
		std::vector<Vector> &centroids, int K) {

	// trivial initialization
	int n, m;
	analyse_dataset(dataset, n, m);
	centroids.resize(K);
	for (auto &p: centroids)
		p.resize(m);

	printf("kmeans++ initializing ...\n");

	// initial cluster with one random point
	Instance2Vector(dataset[random.rand_int() % n], centroids[0], m);

	vector<real_t> distances(n);
	// initialize distances to large numbers
	for (auto &d: distances) d = numeric_limits<real_t>::max();

	Threadpool pool(concurrency);
	int block_size = ceil((double)n / (concurrency));
	for (int k = 1; k < K; k ++) {
		// calculate distances
		vector<future<real_t>> results;
		for (int i = 0, c = 0; i < n; i += block_size, c ++) {
			assert(c < concurrency);
			results.emplace_back(pool.enqueue(bind(update_distances_weighted,
							std::cref(dataset),
							std::cref(centroids[k - 1]),
							std::cref(weight),
							std::ref(distances),
							i, min(i + block_size, n)), 0));
		}

		real_t distsqr_sum = 0;
		for (auto &rst: results)
			distsqr_sum += rst.get();

#ifdef __DEBUG_CHECK
		real_t distsqr_sum_check = 0;
		for (auto &d: distances) distsqr_sum_check += d;
		assert(distsqr_sum_check == distsqr_sum);
#endif

		real_t random_weight = random.rand_int() / (double)RAND_MAX * distsqr_sum;
		for (int i = 0; i < n; i ++) {
			random_weight -= distances[i];
			if (random_weight <= 0) {
				Instance2Vector(dataset[i], centroids[k], m);
				printf("kmeans++ iteration %3d: %f, #%d has been choosen\n", k, distsqr_sum, i); fflush(stdout);
				break;
			}
		}
	}

	return Lloyds_iteration_weighted(dataset, weight, centroids);
}

real_t KMeansppSolver::cluster_weighted(const std::vector<Vector> &dataset, const std::vector<real_t> &weight,
		std::vector<Vector> &centroids, int K) {
	if (dataset.size() == 0) {
		printf("[WARNING] no data given");
		return 0;
	}
	// TODO: not to waste that much memory, write start from scratch
	Dataset inst_dataset(dataset.size());
	for (size_t i = 0; i < dataset.size(); i ++)
		Vector2Instance(dataset[i], inst_dataset[i]);

	return cluster_weighted(inst_dataset, weight, centroids, K);
}
