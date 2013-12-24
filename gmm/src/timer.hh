/*
 * $File: timer.hh
 * $Date: Tue Dec 10 16:44:35 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#pragma once

#include "sys/time.h"
#include <string>

class Timer {
	public:
		long long get_time() {
			timeval tv;
			gettimeofday(&tv, 0);
			return tv.tv_sec * 1000ll + tv.tv_usec / 1000;
		}

		long long m_start;
		long long start() {
			return m_start = get_time();
		}

		long long stop() {
			return get_time() - m_start;
		}
};


class GuardedTimer {
	public:
		std::string prompt;
		Timer timer;
		bool enable;
		GuardedTimer(const std::string &prompt, bool enable = true) :
			prompt(prompt), enable(enable) {
				timer.start();
			}
		~GuardedTimer() {
			if (enable) {
				printf("%s: %.3lfs\n", prompt.c_str(), timer.stop() / 1000.0);
				fflush(stdout);
			}
		}

};

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

