/*
 * $File: fastexp.hh
 * $Date: Tue Dec 10 13:46:58 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#pragma once

void remez5_0_log2_sse(double *values, int num);
void remez7_0_log2_sse(double *values, int num);
void remez9_0_log2_sse(double *values, int num);
void remez11_0_log2_sse(double *values, int num);
void remez5_0_log2(double *values, int num);
void remez7_0_log2(double *values, int num);
void remez9_0_log2(double *values, int num);
void remez11_0_log2(double *values, int num);
void remez13_0_log2(double *values, int num);
void vecexp_remez5_05_05(double *values, int num);
void vecexp_remez7_05_05(double *values, int num);
void vecexp_remez9_05_05(double *values, int num);
void vecexp_remez11_05_05(double *values, int num);
void vecexp_remez13_05_05(double *values, int num);
void vecexp_taylor5(double *values, int num);
void vecexp_taylor7(double *values, int num);
void vecexp_taylor9(double *values, int num);
void vecexp_taylor11(double *values, int num);
void vecexp_taylor13(double *values, int num);
void vecexp_cephes(double *values, int num);

/**
 * vim: syntax=cpp11 foldmethod=marker
 */
