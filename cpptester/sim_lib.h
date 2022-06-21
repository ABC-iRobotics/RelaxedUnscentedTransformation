#pragma once
#include "Stoch.h"

Eigen::VectorXd StateUpdate(const Eigen::VectorXd& x,
  const Eigen::VectorXd& w, double dT);

Eigen::VectorXd TrueOutput(const Eigen::VectorXd& x);

Eigen::VectorXd StateUpdate2(const Eigen::VectorXd& x,
 const Eigen::VectorXd& w, double dT);

Eigen::VectorXd TrueOutput2(const Eigen::VectorXd& x);

UT::ValWithCov RelaxedUT_INES1(double dT, const UT::ValWithCov& x0_in,
  const Eigen::MatrixXd& Sw);

UT::ValWithCov FullUT_INES1(double dT, const UT::ValWithCov& x0,
  const Eigen::MatrixXd& Sw);

UT::ValWithCov RelaxedUT_INES2(const UT::ValWithCov& x0,
  const Eigen::MatrixXd& Sv);

UT::ValWithCov FullUT_INES2(const UT::ValWithCov& x0,
  const Eigen::MatrixXd& Sv);

/*
UT::ValWithCov RelaxedUTN_INES1(double dT, const UT::ValWithCov& x0,
  const Eigen::MatrixXd& Sw, const std::vector<double>& alpha);
UT::ValWithCov FullUTN_INES1(double dT, const UT::ValWithCov& x0,
  const Eigen::MatrixXd& Sw, const std::vector<double>& alpha);
UT::ValWithCov RelaxedUTN_INES2(const UT::ValWithCov& x0,
  const Eigen::MatrixXd& Sv, const std::vector<double>& alpha);
UT::ValWithCov FullUTN_INES2(const UT::ValWithCov& x0,
  const Eigen::MatrixXd& Sv, const std::vector<double>& alpha);*/
