/*
 * $File: kmeansII.cc
 * $Date: Sun Dec 15 20:23:59 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#include "kmeansII.hh"
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

KMeansIISolver::KMeansIISolver(int concurrency, real_t oversampling_factor, real_t size_factor) :
	KMeansSolver(concurrency), oversampling_factor(oversampling_factor), size_factor(size_factor) {
}

static void Instance2Vector(const Instance &instance, Vector &ret, int dim) {
	ret.resize(dim);
	for (auto &item: instance)
		ret[item.first] = item.second;
}

namespace KMeansIISolverImpl {

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
			const vector<Vector> &centroids,
			int centroid_begin, int centroid_end,
			vector<real_t> &distances,
			vector<int> &belong,
			int begin, int end) {

		real_t distsqr_sum = 0;

		for (int i = begin; i < end; i ++) {
			const Instance &inst = dataset[i];
			real_t &min_distsqr = distances[i];
			for (int j = centroid_begin; j < centroid_end; j ++) {
				real_t distsqr = distancesqr(inst, centroids[j]);
				if (distsqr < min_distsqr) {
					min_distsqr = distsqr;
					belong[i] = j;
				}
			}
			distsqr_sum += min_distsqr;
		}
		return distsqr_sum;
	}
}

using namespace KMeansIISolverImpl;

real_t KMeansIISolver::cluster(const Dataset &dataset, std::vector<Vector> &centroids_param, int K) {
	// trivial initialization
	int n, m;
	analyse_dataset(dataset, n, m);
	centroids_param.resize(K);
	for (auto &p: centroids_param)
		p.resize(m);

	printf("kmeansII initializing ...\n");
	printf("concurrency II: ######## %d\n", concurrency);

	// initial cluster with one random point
	std::vector<Vector> centroids(1, centroids_param[0]);
	Instance2Vector(dataset[rand() % n], centroids[0], m);

	vector<real_t> distances(n);
	vector<int> belong(n);
	// initialize distances to large numbers
	for (auto &d: distances) d = numeric_limits<real_t>::max();


	int block_size = ceil((double)n / (concurrency));

	// guard for thread pool
	{
		int last_size = 0;
		Threadpool pool(concurrency);
		int iter_id = 0;
		while (true) {
			// calculate distances
			vector<future<real_t>> results;
			for (int i = 0, c = 0; i < n; i += block_size, c ++) {
				assert(c < concurrency);

				results.emplace_back(pool.enqueue(bind(update_distances,
								std::cref(dataset),
								std::cref(centroids),
								last_size, centroids.size(),
								std::ref(distances),
								std::ref(belong),
								i, min(i + block_size, n)), 0));
			}
			if (centroids.size() > size_factor * K)
				break;

			real_t distsqr_sum = 0;
			for (auto &rst: results)
				distsqr_sum += rst.get();

#ifdef __DEBUG_CHECK
			real_t distsqr_sum_check = 0;
			for (auto &d: distances) distsqr_sum_check += d;
			assert(distsqr_sum_check == distsqr_sum);
#endif
			last_size = centroids.size();

			for (int i = 0; i < n; i ++) {
				real_t random_weight = rand() / (double)RAND_MAX * distsqr_sum;
				if (random_weight < distances[i] * oversampling_factor * K) {
					centroids.emplace_back(Vector());
					Instance2Vector(dataset[i], centroids.back(), m);
				}
			}

			printf("kmeansII iteration %3d: %f, #new: %lu, #all: %lu\n",
					iter_id, distsqr_sum, centroids.size() - last_size, centroids.size());
			fflush(stdout);

			if (centroids.size() - last_size == 0)
				break;
			iter_id ++;
		}

	}

	while (centroids.size() <= size_factor * K) {
		centroids.emplace_back(Vector());
		Instance2Vector(dataset[random.rand_int(n)], centroids.back(), m);
	}
	// calculate weight
	vector<real_t> weight(centroids.size());
	for (auto &bl: belong)
		weight[bl] += 1.0;

	// weighted clustering
	KMeansppSolver kmeanspp_solver(concurrency);
	kmeanspp_solver.cluster_weighted(centroids, weight, centroids_param, K);

	return Lloyds_iteration(dataset, centroids_param);
}

