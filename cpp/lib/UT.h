// Traditional and selective Unscented Transformation implementations

#pragma once

#include "Eigen/Dense"
#include <Eigen/SparseCore>
#include "index_selector.h"
#include "PartialCholevski.h"

// Selective UT in indices contained by vector "inl"
template<typename Func>
void SelUT(const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
	const Eigen::VectorXi& inl, Func fin, Eigen::VectorXd& z,
	Eigen::MatrixXd& Sz, Eigen::MatrixXd& Sxz);

// General UT
template<typename Func>
void UT(const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
	Func fin, Eigen::VectorXd& z, Eigen::MatrixXd& Sz, Eigen::MatrixXd& Sxz);

// Selective UT in indices 0..(m-1) and applying back rotation via Q.transpose
template<typename Func>
void SelUT(const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0, int m,
	Func fin, Eigen::VectorXd& z, Eigen::MatrixXd& Sz, Eigen::MatrixXd& Sxz,
	Eigen::SparseMatrix<double> Q = Eigen::SparseMatrix<double>(0, 0));


template<typename Func>
void SelUT(const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
	const Eigen::VectorXi& inl, Func fin, Eigen::VectorXd& z,
	Eigen::MatrixXd& Sz, Eigen::MatrixXd& Sxz) {
	// 
	int m = (int)inl.size();
	int n = (int)x0.size();
	// nonlinear dependency vector
	Eigen::VectorXi NL = Eigen::VectorXi::Zero(n);
	for (int i = 0; i < inl.size(); i++)
		NL[inl[i]] = 1;
	// weights
	//double alpha = 0.7;
	//double lambda = alpha * alpha * m - m;
	double lambda = 3 - m;
	if (lambda < 0)
		lambda = 0;
	double nl = m + lambda;
	auto sqrtnl = sqrt(nl);
	double W0 = lambda / nl;
	//double W0cov = W0 + 1 + 2 - alpha * alpha;
	double W0cov = W0;
	double Wi = 0.5 / nl;
	// partial Choleski
	Eigen::MatrixXd L = SF::PartialChol(S0, NL);
	L *= sqrtnl;
	// Map the sigma points
	std::vector<Eigen::VectorXd> Zi;
	Zi.push_back(fin(x0));
	for (int k = 0; k < m; k++)
		Zi.push_back(fin(x0 - L.col(k)));
	for (int k = 0; k < m; k++)
		Zi.push_back(fin(x0 + L.col(k)));
	int g = (int)Zi[0].size();
	// Expected z value
	z = Eigen::VectorXd::Zero(g);
	for (int k = 1; k < Zi.size(); k++)
		z += Zi[k];
	z *= Wi;
	z += W0 * Zi[0];
	// Sigma_z
	Sz = Eigen::MatrixXd::Zero(g, g);
	{
		for (int k = 1; k < 2 * m + 1; k++) {
			auto temp = Zi[k] - z;
			Sz += temp * temp.transpose();
			//std::cout << Sz << std::endl << std::endl;
		}
		Sz *= Wi;
		auto temp = Zi[0] - z;
		Sz += W0cov * temp*temp.transpose();
		//std::cout << Sz << std::endl << std::endl;
	}
	// Sigma _zx
	Sxz = Eigen::MatrixXd::Zero(n, g);
	{
		for (int k = 0; k < m; k++)
			Sxz += L.block(0, k, n, 1)* (Zi[k + 1 + m] - Zi[k + 1]).transpose();
		Sxz *= Wi;
	}
}

template<typename Func>
void UT(const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0,
	Func fin, Eigen::VectorXd& z, Eigen::MatrixXd& Sz, Eigen::MatrixXd& Sxz) {
	SelUT(x0, S0, (int)x0.size(), fin, z, Sz, Sxz);
}

// x = [ xnl:R^m  xl:R^(n-m) ]
template<typename Func>
void SelUT(const Eigen::VectorXd& x0, const Eigen::MatrixXd& S0, int m,
	Func fin, Eigen::VectorXd& z, Eigen::MatrixXd& Sz, Eigen::MatrixXd& Sxz,
	Eigen::SparseMatrix<double> Q) {
	int n = (int)x0.size();
	// nonlinear dependency vector
	Eigen::VectorXi NL = Eigen::VectorXi::Ones(n);
	for (int k = m; k < n; k++)
		NL[k] = 0;
	// weights
	double alpha = 0.7;
	double lambda = alpha * alpha * m - m;
	double nl = m + lambda;
	auto sqrtnl = sqrt(nl);
	double W0 = lambda / nl;
	double W0cov = W0 + 1 + 2 - alpha * alpha;
	double Wi = 0.5 / nl;
	// partial Choleski
	Eigen::MatrixXd L = SF::PartialChol(S0, NL);
	if (Q.rows() > 0)
		L = Q.transpose() * L;
	L *= sqrtnl;
	// Map the sigma points
	std::vector<Eigen::VectorXd> Zi;
	Zi.push_back(fin(x0));
	for (int k = 0; k < m; k++)
		Zi.push_back(fin(x0 - L.col(k)));
	for (int k = 0; k < m; k++)
		Zi.push_back(fin(x0 + L.col(k)));
	//for (int i = 0; i < Zi.size(); i++)
	//	std::cout << Zi[i].transpose() << std::endl << std::endl;
	int g = (int)Zi[0].size();
	// Expected z value
	z = Eigen::VectorXd::Zero(g);
	for (int k = 1; k < Zi.size(); k++)
		z += Zi[k];
	z *= Wi;
	z += W0 * Zi[0];
	// Sigma_z
	Sz = Eigen::MatrixXd::Zero(g, g);
	{
		for (int k = 1; k < 2 * m + 1; k++) {
			auto temp = Zi[k] - z;
			Sz += temp * temp.transpose();
		}
		Sz *= Wi;
		auto temp = Zi[0] - z;
		Sz += W0cov * temp*temp.transpose();
	}
	// Sigma _zx
	Sxz = Eigen::MatrixXd::Zero(n, g);
	{
		for (int k = 0; k < m; k++)
			Sxz += L.block(0, k, n, 1)* (Zi[k + 1 + m] - Zi[k + 1]).transpose();
		Sxz *= Wi;
	}
}
