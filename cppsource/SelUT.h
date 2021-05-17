#pragma once
#include "Eigen/Dense"
#include <Eigen/SparseCore>
#include "index_selector.h"
#include "PartialCholevski.h"

namespace RelaxedUnscentedTransformation {

	/*! \brief Selective Unscented transformation
	*
	* Considering a function z=f(x), where f depends only on values of x with given indices,
	* the algorithm approximates the expected value of z,
	* its covariance matrix Sz, and cross covariance matrix Sxz
	* from the expected value of x and its covariance matrix Sx
	*
	* Inputs:
	*   x: a vector of size n
	*   inl: indices of x, the function depends on
	*   Sx: a matrix (symmetric, poisitive definite) of size n x n
	*   fin: a function  y=fin(x) (that can be a C callback, std::function, etc.)
	*
	* Outputs:
	*   z: a vector of size g
	*   Sz: a matrix of size g x g
	*  Sxz: a matrix of size n x g
	*/
	template<typename Func>
	void SelUT(const Eigen::VectorXd& x, const Eigen::MatrixXd& Sx,
		const Eigen::VectorXi& inl, Func fin, Eigen::VectorXd& z,
		Eigen::MatrixXd& Sz, Eigen::MatrixXd& Sxz);

	/*! \brief Selective Unscented transformation
	*
	* Considering a function z=f(x), where f depends only on the first m values of vector (Q*x),
	* the algorithm approximates the expected value of z,
	* its covariance matrix Sz, and cross covariance matrix Sxz
	* from the expected value of x and its covariance matrix Sx
	*
	* Inputs:
	*   x: a vector of size n
	*   Q: an orthogonal matrix of size n x n (default value: identity matrix)
	*   m: num. of variables the function depends on (<=n)
	*   Sx: a matrix (symmetric, poisitive definite) of size n x n
	*   fin: a function  y=fin(x) (that can be a C callback, std::function, etc.)
	*
	* Outputs:
	*   z: a vector of size g
	*   Sz: a matrix of size g x g
	*  Sxz: a matrix of size n x g
	*/
	template<typename Func>
	void SelUT(const Eigen::VectorXd& x, const Eigen::MatrixXd& Sx, int m,
		Func fin, Eigen::VectorXd& z, Eigen::MatrixXd& Sz, Eigen::MatrixXd& Sxz,
		Eigen::SparseMatrix<double> Q = Eigen::SparseMatrix<double>(0, 0));

	template<typename Func>
	void SelUT(const Eigen::VectorXd& x, const Eigen::MatrixXd& Sx,
		const Eigen::VectorXi& inl, Func fin, Eigen::VectorXd& z,
		Eigen::MatrixXd& Sz, Eigen::MatrixXd& Sxz) {
		// 
		int m = (int)inl.size();
		int n = (int)x.size();
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
		Eigen::MatrixXd L = PartialChol(Sx, NL);
		L *= sqrtnl;
		// Map the sigma points
		std::vector<Eigen::VectorXd> Zi;
		Zi.push_back(fin(x));
		for (int k = 0; k < m; k++)
			Zi.push_back(fin(x - L.col(k)));
		for (int k = 0; k < m; k++)
			Zi.push_back(fin(x + L.col(k)));
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

	template<typename Func>
	void SelUT(const Eigen::VectorXd& x, const Eigen::MatrixXd& Sx, int m,
		Func fin, Eigen::VectorXd& z, Eigen::MatrixXd& Sz, Eigen::MatrixXd& Sxz,
		Eigen::SparseMatrix<double> Q) {
		int n = (int)x.size();
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
		Eigen::MatrixXd L = PartialChol(Sx, NL);
		if (Q.rows() > 0)
			L = Q.transpose() * L;
		L *= sqrtnl;
		// Map the sigma points
		std::vector<Eigen::VectorXd> Zi;
		Zi.push_back(fin(x));
		for (int k = 0; k < m; k++)
			Zi.push_back(fin(x - L.col(k)));
		for (int k = 0; k < m; k++)
			Zi.push_back(fin(x + L.col(k)));
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

}
