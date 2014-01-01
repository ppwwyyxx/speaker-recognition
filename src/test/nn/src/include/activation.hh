/*
 * $File: activation.hh
 * $Date: Thu Oct 17 11:02:51 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#pragma once

#include "type.hh"
#include <cmath>

/**
 * Activation Function
 */
class ActivationFunction
{
	public:

		// `squash' the input
		virtual real_t value_at(real_t x) = 0;

		// derivative at x
		virtual real_t derivative(real_t x) = 0;

		/// derivative at x, assuming value_at(x) == y
		virtual real_t derivative(real_t x, real_t y) = 0;

		virtual ~ActivationFunction() {}
};

class ActivationLogistic : public ActivationFunction
{
	public:
		// `squash' the input
		real_t alpha;
		ActivationLogistic() {
			alpha = 1.0;
		}
		virtual real_t value_at(real_t x) {
			if (x < -700.0) // avoid triggering a floating point exception
				return 0.0;
			if (x >= 700.0) // avoid triggering a floating point exception
				return 1.0;
			return 1.0 / (exp(-alpha * x) + 1.0);
		}

		// derivative at x
		virtual real_t derivative(real_t x) {
			real_t y = this->value_at(x);
			return alpha * y * (1 - y);
		}

		/// derivative at x, assuming value_at(x) == y
		virtual real_t derivative(real_t, real_t y) {
			return alpha * y * (1 - y);
		}
};

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

