/*
 * $File: random.hh
 * $Date: Wed Dec 11 17:31:04 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#pragma once

#include "type.hh"

#include <random>
#include <chrono>
#include <limits>

/**
 * Random Number Generator
 */
class Random {
	public:

		Random() {
//            long long seed = std::chrono::system_clock::now().time_since_epoch().count();
			long long seed = rand();
			generator.seed(seed);
		}

		void seed(unsigned long long s) {
			generator.seed(s);
		}

		/// return a random number in [0.0, 1.0)
		inline real_t rand_real() {
			return real_distribution(generator);
		}

		inline real_t rand_normal() {
			return normal_distribution(generator);
		}

		inline real_t rand_normal(real_t mean, real_t stddev) {
			return std::normal_distribution<real_t>(mean, stddev)(generator);
		}

		void set_normal_distribution_param(real_t mean, real_t stddev) {
			normal_distribution = std::normal_distribution<real_t>(mean, stddev);
		}

		/// return a random integer in range [0, max_val)
		inline int rand_int(int max_val = std::numeric_limits<int>::max()) {
			return rand_real() * max_val;
		}

	protected:
		std::default_random_engine generator;
		std::uniform_real_distribution<real_t> real_distribution; // [0.0, 1.0)
		std::normal_distribution<real_t> normal_distribution; // N(0, 1)
};

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

