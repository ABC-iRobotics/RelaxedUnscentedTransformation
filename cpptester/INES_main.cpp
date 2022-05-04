#include "INES_model.h"
#include <iostream>

using namespace UT_INES;
using namespace Eigen;

#include <chrono>
#include "FilterRMSE.h"

const double Ts = 0.05;

const bool useHO = true;

int main() {
  Model2DAbsRel model;
  auto RMSE_ = RMSE(Ts, 100, 1000, { {true,useHO,{1}} , {true,useHO,{1.5,0.75}} }, & model);

  for (auto it: RMSE_)
	std::cout << it << std::endl;
	
  return 0;
}