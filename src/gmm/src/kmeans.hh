/*
 * $File: kmeans.hh
 * $Date: Wed Dec 11 19:03:51 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#pragma once
#include "dataset.hh"

/**
 * A KMeans clustering algorithm
 */
class KMeansSolver {
	public:
		/**
		 * number of parallel workers.
		 * TODO:
		 * @threshold_relative stop when the ratio between cost reduction
		 *		and cost is less than @threshold_relative, default 0.0001
		 */
		KMeansSolver(int concurrency);

		/**
		 * cluster point in dataset into K partitions
		 * situation when K <= dataset.size() should be correctly dealt with
		 */
		virtual real_t cluster(const Dataset &dataset, std::vector<Vector> &centroids, int K);

		/**
		 * iteration step
		 */
		real_t Lloyds_iteration(const Dataset &dataset, std::vector<Vector> &centroids);
		real_t Lloyds_iteration_weighted(const Dataset &dataset, const std::vector<real_t> &weight,
				std::vector<Vector> &centroids);

	protected:
		int concurrency;
};


/**
 * vim: syntax=cpp11 foldmethod=marker
 */

