#pragma once

#include "UTComponents.h"
#include <iostream>

namespace UT_INES {
  Eigen::VectorXd StateUpdate(const Eigen::VectorXd& x, const Eigen::VectorXd& w, double dT);

  Eigen::VectorXd TrueOutput(const Eigen::VectorXd& x);

  Eigen::VectorXd StateUpdate2(const Eigen::VectorXd& x, const Eigen::VectorXd& w, double dT);

  Eigen::VectorXd TrueOutput2(const Eigen::VectorXd& x);

  ValWithCov RelaxedUT_INES1(double dT, const ValWithCov& x0_in,
	const Eigen::MatrixXd& Sw);

  ValWithCov RelaxedUTN_INES1(double dT, const ValWithCov& x0,
	const Eigen::MatrixXd& Sw, const std::vector<double>& alpha);

  ValWithCov FullUT_INES1(double dT, const ValWithCov& x0,
	const Eigen::MatrixXd& Sw);

  ValWithCov FullUTN_INES1(double dT, const ValWithCov& x0,
	const Eigen::MatrixXd& Sw, const std::vector<double>& alpha);

  ValWithCov RelaxedUT_INES2(const ValWithCov& x0,
	const Eigen::MatrixXd& Sv);

  ValWithCov RelaxedUTN_INES2(const ValWithCov& x0,
	const Eigen::MatrixXd& Sv, const std::vector<double>& alpha);

  ValWithCov FullUT_INES2(const ValWithCov& x0,
	const Eigen::MatrixXd& Sv);

  ValWithCov FullUTN_INES2(const ValWithCov& x0,
	const Eigen::MatrixXd& Sv, const std::vector<double>& alpha);
}
