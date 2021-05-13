// Traditional and selective Unscented Transformation implementations

#pragma once
#include <iostream>
#include "Eigen/Dense"
#include "PartialCholevski.h"

// MNMPES
template<typename Func>
void AUT(const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
	Func fin, const Eigen::VectorXd& zmeas, double sfmin, double sfmax,
	double dsf, Eigen::VectorXd& z, Eigen::MatrixXd& Sz, Eigen::MatrixXd& Sxz) {
	//
	int n = (int)x0.size();
	// nonlinear dependency vector
	Eigen::VectorXi NL = Eigen::VectorXi::Zero(n);
	for (int i = 0; i < n; i++)
		NL[i] = 1;
	// partial Choleski
	Eigen::MatrixXd L = SF::PartialChol(S0, NL);
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
		Zi.push_back(fin(x0));
		for (int k = 0; k < n; k++)
			Zi.push_back(fin(x0 - scaledL.col(k)));
		for (int k = 0; k < n; k++)
			Zi.push_back(fin(x0 + scaledL.col(k)));
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
				//std::cout << Sz << std::endl << std::endl;
			}
			Sznew *= Wi;
			auto temp = Zi[0] - znew;
			Sznew += W0cov * temp*temp.transpose();
			//std::cout << Sz << std::endl << std::endl;
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
