/*
 * $File: dataset.hh
 * $Date: Sun Sep 08 08:47:09 2013 +0800
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#pragma once

#include "type.hh"

#include <vector>

typedef std::vector<std::pair<int, real_t> >	Instance;
typedef std::vector<Instance>					Dataset;
typedef std::vector<Instance *>					RefDataset;
typedef std::vector<const Instance *>			ConstRefDataset;
typedef std::vector<int>						Labels;
typedef std::vector<real_t>						RealLabels;

typedef std::vector<real_t>						Vector;

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

