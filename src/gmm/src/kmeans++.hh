/*
 * $File: kmeans++.hh
 * $Date: Wed Dec 11 17:36:01 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#pragma once
#include "dataset.hh"
#include "kmeans.hh"
#include "random.hh"

/**
 * A KMeans++ clustering algorithm
 */
class KMeansppSolver : public KMeansSolver {
	public:
		/**
		 * number of parallel workers.
		 */
		KMeansppSolver(int concurrency);

		/**
		 * cluster point in dataset into K partitions
		 * situation when K <= dataset.size() should be correctly dealt with
		 */
		virtual real_t cluster(const Dataset &dataset, std::vector<Vector> &centroids, int K);
		virtual real_t cluster_weighted(const Dataset &dataset, const std::vector<real_t> &weight,
				std::vector<Vector> &centroids, int K);
		virtual real_t cluster_weighted(const std::vector<Vector> &dataset, const std::vector<real_t> &weight,
				std::vector<Vector> &centroids, int K);

		Random random;

};


/**
 * vim: syntax=cpp11 foldmethod=marker
 */

