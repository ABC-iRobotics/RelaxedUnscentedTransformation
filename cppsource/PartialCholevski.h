#pragma once
#include "Eigen/Dense"

namespace RelaxedUnscentedTransformation {

	/* \brief Cholevski-factorization of positive semi-definite matrices in the given columns
	 *
	 * Based on the Cholevski-Crout algrithm
	 */
	Eigen::MatrixXd PartialChol(const Eigen::MatrixXd& a, const Eigen::VectorXi& v);
}