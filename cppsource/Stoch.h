#ifndef STOCH_H
#define STOCH_H
#include "Eigen/Dense"

namespace UT {

  struct ValWithCov {
	Eigen::VectorXd y;
	Eigen::MatrixXd Sy, Sxy;
	ValWithCov(const Eigen::VectorXd y_, const Eigen::MatrixXd& Sy_,
	  const Eigen::MatrixXd& Sxy_);;
	ValWithCov(const Eigen::VectorXd y_, const Eigen::MatrixXd& Sy_);;
  };

  Eigen::VectorXd RandomVector(const Eigen::MatrixXd& S);
}

#endif