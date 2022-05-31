#pragma once
#include "Eigen/Dense"

namespace UT {

  Eigen::MatrixXd PartialChol(const Eigen::MatrixXd& a, const Eigen::VectorXi& inl);

  Eigen::MatrixXd FullChol(const Eigen::MatrixXd& a);

}