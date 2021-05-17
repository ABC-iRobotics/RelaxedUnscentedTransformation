#pragma once

#include <iostream>
#include "Eigen/Dense"
#include "PartialCholevski.h"

// no g and no Q
template <typename Func>
void RelAUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il,
	Func fin, const Eigen::MatrixXd& F, const Eigen::VectorXi& inl,
	const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
	const Eigen::VectorXd& ymeas, double sfmin, double sfmax, double dsf,
	Eigen::VectorXd& y, Eigen::MatrixXd& Sy, Eigen::MatrixXd& Sxy) {
	// Perform the scaling factor independent computations
	// Init x_l
	Eigen::VectorXd xl = VectorSelect(x0, il);
	auto At = A.transpose();
	Eigen::MatrixXd Sxy0 = MatrixColumnSelect(S0, il) * At;
	int n = (int)x0.size();
	int m = (int)inl.size();
	// nonlinear dependency vector
	Eigen::VectorXi NL = Eigen::VectorXi::Zero(n);
	for (int i = 0; i < inl.size(); i++)
		NL[inl[i]] = 1;
	// partial Choleski
	Eigen::MatrixXd L = PartialChol(S0, NL);
	// Main iteration
	double bestcost, lambdabest;
	for (double lambda = sfmin; lambda <= sfmax; lambda += dsf) {
		// Perform UT around a
		// weights
		double nl = m + lambda;
		auto sqrtnl = sqrt(nl);
		double W0 = lambda / nl;
		//double W0cov = W0 + 1 + 2 - alpha * alpha;
		double W0cov = W0;
		double Wi = 0.5 / nl;
		Eigen::MatrixXd scaledL = L * sqrtnl;
		// Map the sigma points
		std::vector<Eigen::VectorXd> Zi;
		Zi.push_back(fin(x0));
		for (int k = 0; k < m; k++)
			Zi.push_back(fin(x0 - scaledL.col(k)));
		for (int k = 0; k < m; k++)
			Zi.push_back(fin(x0 + scaledL.col(k)));
		int g = (int)Zi[0].size();
		// Expected z value
		Eigen::VectorXd b0 = Eigen::VectorXd::Zero(g);
		for (int k = 1; k < Zi.size(); k++)
			b0 += Zi[k];
		b0 *= Wi;
		b0 += W0 * Zi[0];
		// Sigma_z
		Eigen::MatrixXd Sb0 = Eigen::MatrixXd::Zero(g, g);
		{
			for (int k = 1; k < 2 * m + 1; k++) {
				auto temp = Zi[k] - b0;
				Sb0 += temp * temp.transpose();
				//std::cout << Sz << std::endl << std::endl;
			}
			Sb0 *= Wi;
			auto temp = Zi[0] - b0;
			Sb0 += W0cov * temp*temp.transpose();
			//std::cout << Sz << std::endl << std::endl;
		}
		// Sigma _zx
		Eigen::MatrixXd Sxb0 = Eigen::MatrixXd::Zero(n, g);
		{
			for (int k = 0; k < m; k++)
				Sxb0 += scaledL.block(0, k, n, 1)* (Zi[k + 1 + m] - Zi[k + 1]).transpose();
			Sxb0 *= Wi;
		}
		// Determine b related quantities
		Eigen::VectorXd b(b0.size() + F.rows());
		b.segment(0, b0.size()) = b0;
		b.segment(b0.size(), F.rows()) = F * b0;
		Eigen::MatrixXd Sb(b.size(), b.size());
		Sb.block(0, 0, b0.size(), b0.size()) = Sb0;
		{
			auto temp = F * Sb0;
			Sb.block(b0.size(), 0, F.rows(), b0.size()) = temp;
			Sb.block(0, b0.size(), b0.size(), F.rows()) = temp.transpose();
			Sb.block(b0.size(), b0.size(), F.rows(), F.rows()) = temp * F.transpose();
		}
		Eigen::MatrixXd Sxb(x0.size(), b.size());
		Sxb.block(0, 0, x0.size(), b0.size()) = Sxb0;
		Sxb.block(0, b0.size(), x0.size(), F.rows()) = Sxb0 * F.transpose();
		// Determine y related quantities
		Eigen::VectorXd ynew = b + A * xl;
		Eigen::MatrixXd Sxynew = Sxb + Sxy0;
		Eigen::MatrixXd Synew = Sb + A * MatrixRowSelect(Sxynew, il) + MatrixColumnSelect(Sxb.transpose(), il)*At;
		Eigen::VectorXd dy = ynew - ymeas;
		for (int i = 0; i < dy.size() / 2; i++) {
			if (dy(2 * i + 1) < -EIGEN_PI)
				dy(2 * i + 1) += 2. * EIGEN_PI;
			if (dy(2 * i + 1) > -EIGEN_PI)
				dy(2 * i + 1) -= 2. * EIGEN_PI;
		}
		double newcost = dy.transpose() * Synew.inverse() * dy;
		if (lambda == sfmin || newcost < bestcost) {
			bestcost = newcost;
			y = ynew;
			Sy = Synew;
			Sxy = Sxynew;
			lambdabest = lambda;
		}
	}
}
