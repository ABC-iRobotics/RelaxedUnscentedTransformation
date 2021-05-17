#pragma once
#include "SelUT.h"

namespace RelaxedUT {

	/*! \brief Traditional Unscented transformation
	*
	* Considering a function z=f(x), the algorithm approximates the expected value of z,
	* its covariance matrix Sz, and cross covariance matrix Sxz
	* from the expected value of x and its covariance matrix Sx
	*
	* Inputs:
	*   x: a vector of size n
	*   Sx: a matrix (symmetric, poisitive definite) of size n x n
	*   fin: a function  z=fin(x) (that can be a C callback, std::function, etc.)
	*
	* Outputs:
	*   z: a vector of size g
	*   Sz: a matrix of size g x g
	*  Sxz: a matrix of size n x g
	*/
	template<typename Func>
	void UT(const Eigen::VectorXd& x, const Eigen::MatrixXd& Sx,
		Func fin, Eigen::VectorXd& z, Eigen::MatrixXd& Sz, Eigen::MatrixXd& Sxz);

	template<typename Func>
	void UT(const Eigen::VectorXd& x, const Eigen::MatrixXd& Sx,
		Func fin, Eigen::VectorXd& z, Eigen::MatrixXd& Sz, Eigen::MatrixXd& Sxz) {
		SelUT(x, Sx, (int)x.size(), fin, z, Sz, Sxz);
	}
}
