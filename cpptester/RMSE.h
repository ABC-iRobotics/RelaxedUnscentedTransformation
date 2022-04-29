﻿#ifndef RMSE_H
#define RMSE_H

#include "SelUT.h"

double RMSE(int N, int K, const RelaxedUnscentedTransformation::UTSettings& settings);

double RMSE_sep(int N, int K, const RelaxedUnscentedTransformation::UTSettings& settings);

#endif