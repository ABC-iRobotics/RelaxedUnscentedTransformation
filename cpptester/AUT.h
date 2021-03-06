#pragma once
#include "PartialCholevski.h"

/* \brief Adaptive extension of traditional UT according to MNMPES criteria
*
* The MNMPES criteria can be found in paper:
* J. Dunik, M. Simandl, and O. Straka, “Unscented Kalman filter: aspects
* and adaptive setting of scaling parameter,” IEEE Transactions on
* Automatic Control, vol. 57, no. 9, pp. 2411–2416, 2012
*
* Syntax:
*  Considering a function z = f(x), the algorithm approximates the expected value of z,
* its covariance matrix Sz, and cross covariance matrix Sxz
* from the expected value of x and its covariance matrix Sx
*
* Inputs:
*	x : a vector of size n
*   Sx : a matrix(symmetric, poisitive definite) of size n x n
*   fin : a function  z = fin(x) (that can be a C callback, std::function, etc.)
*	zmeas : the measured output of the function
*	sfmin, sfmax, dsf : optimal scaling factor is searched on the grid [sfmin:dsf:sfmax]
*
* Outputs :
*	z : a vector of size g
*   Sz : a matrix of size g x g
*  Sxz : a matrix of size n x g
*/
template<typename Func>
void AUT(const Eigen::VectorXd& x, const Eigen::MatrixXd& Sx,
	Func fin, const Eigen::VectorXd& zmeas, double sfmin, double sfmax,
	double dsf, Eigen::VectorXd& z, Eigen::MatrixXd& Sz, Eigen::MatrixXd& Sxz) {
	//
	int n = (int)x.size();
	// nonlinear dependency vector
	Eigen::VectorXi NL = Eigen::VectorXi::Zero(n);
	for (int i = 0; i < n; i++)
		NL[i] = 1;
	// partial Choleski
	Eigen::MatrixXd L = PartialChol(Sx, NL);
	double bestcost, lambdabest;
	for (double lambda = sfmin; lambda <= sfmax; lambda += dsf) {
		// weights
		double nl = n + lambda;
		auto sqrtnl = sqrt(nl);
		double W0 = lambda / nl;
		double W0cov = W0;
		double Wi = 0.5 / nl;
		// scale L
		Eigen::MatrixXd scaledL = L * sqrtnl;
		// Map the sigma points
		std::vector<Eigen::VectorXd> Zi;
		Zi.push_back(fin(x));
		for (int k = 0; k < n; k++)
			Zi.push_back(fin(x - scaledL.col(k)));
		for (int k = 0; k < n; k++)
			Zi.push_back(fin(x + scaledL.col(k)));
		int g = (int)Zi[0].size();
		// Expected z value
		Eigen::VectorXd znew = Eigen::VectorXd::Zero(g);
		for (int k = 1; k < Zi.size(); k++)
			znew += Zi[k];
		znew *= Wi;
		znew += W0 * Zi[0];
		// Sigma_z
		Eigen::MatrixXd Sznew = Eigen::MatrixXd::Zero(g, g);
		{
			for (int k = 1; k < 2 * n + 1; k++) {
				auto temp = Zi[k] - znew;
				Sznew += temp * temp.transpose();
			}
			Sznew *= Wi;
			auto temp = Zi[0] - znew;
			Sznew += W0cov * temp*temp.transpose();
		}
		Eigen::VectorXd dz = znew - zmeas;
		for (int i = 0; i < dz.size() / 2; i++) {
			if (dz(2 * i + 1) < -EIGEN_PI)
				dz(2 * i + 1) += 2. * EIGEN_PI;
			if (dz(2 * i + 1) > -EIGEN_PI)
				dz(2 * i + 1) -= 2. * EIGEN_PI;
		}
		double newcost = dz.transpose() * Sznew.inverse() * dz;
		if (lambda == sfmin || newcost < bestcost) {
			bestcost = newcost;
			z = znew;
			Sz = Sznew;
			lambdabest = lambda;
			// Sigma _zx
			Sxz = Eigen::MatrixXd::Zero(n, g);
			for (int k = 0; k < n; k++)
				Sxz += scaledL.block(0, k, n, 1)* (Zi[k + 1 + n] - Zi[k + 1]).transpose();
			Sxz *= Wi;
		}
	}
	//std::cout << "Used scaling factor: " << lambdabest << std::endl << std::endl;
}
