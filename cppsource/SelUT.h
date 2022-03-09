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
	*   N: order of approximation (2,..,4 is allowed)
	*
	* Outputs:
	*   z: a vector of size g
	*   Sz: a matrix of size g x g
	*  Sxz: a matrix of size n x g
	*/
	template<typename Func>
	void SelUT(const Eigen::VectorXd& x, const Eigen::MatrixXd& Sx,
	  const Eigen::VectorXi& inl, Func fin, int N,
	  Eigen::VectorXd& z, Eigen::MatrixXd& Sz, Eigen::MatrixXd& Sxz);

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
	*   N: order of approximation (2,..,4 is allowed)
	*
	* Outputs:
	*   z: a vector of size g
	*   Sz: a matrix of size g x g
	*  Sxz: a matrix of size n x g
	*/
	template<typename Func>
	void SelUT(const Eigen::VectorXd& x, const Eigen::MatrixXd& Sx, int m,
	  Func fin, int N, Eigen::VectorXd& z, Eigen::MatrixXd& Sz, Eigen::MatrixXd& Sxz,
	  Eigen::SparseMatrix<double> Q = Eigen::SparseMatrix<double>(0, 0));


	//////////////////////////////////////////////////////////////////////////////////
	//////// Implementations
	//////////////////////////////////////////////////////////////////////////////////

	inline double kappa_abs(int N, int l) {
	  switch (N) {
	  case 2:
		if (l == 0)
		  return 3;
		break;
	  case 3:
		switch (l) {
		case 0:
		  return 5.45;
		case 1:
		  return 0.55;
		}
	  case 4:
		switch (l) {
		case 0:
		  return 6.894;
		case 1:
		  return 1.8658;
		case 2:
		  return 1.8658;
		}
	  }
	  throw std::runtime_error("kappa_abs(): Not allowed inputs");
	  return 0;
	}

	template<typename Func>
	void SelUTCore(const Eigen::VectorXd& x, const Eigen::MatrixXd& L,
	  Func fin, int N, Eigen::VectorXd& z, Eigen::MatrixXd& Sz,
	  Eigen::MatrixXd& Sxz) {
	  int m = L.cols();
	  int n = L.rows();
	  // Scale offsets
	  std::vector<Eigen::MatrixXd> Lscaled;
	  for (int l = 0; l < N - 1; l++)
		Lscaled.push_back(L * sqrt(kappa_abs(N, l)));
	  // Map the sigma points
	  std::vector<std::vector<Eigen::VectorXd>> Zi;
	  Zi.push_back({ fin(x) });
	  for (int l = 0; l < N - 1; l++) {
		std::vector<Eigen::VectorXd> Zi_l;
		for (int b = 0; b < m; b++)
		  Zi_l.push_back(fin(x - Lscaled[l].col(b)));
		for (int b = 0; b < m; b++)
		  Zi_l.push_back(fin(x + Lscaled[l].col(b)));
		Zi.push_back(Zi_l);
	  }
	  // Compute weights
	  Eigen::VectorXd W(N);
	  W[0] = 0;
	  for (int l = 1; l < N; l++) // W0 will be computed later
		W[l] = 0.5 / (double(N) - 1) / kappa_abs(N, l-1);
	  W[0] = 1. - W.sum() * 2. * double(m);
	  // Expected z value
	  int g = (int)Zi[0][0].size();
	  z = W[0] * Zi[0][0];
	  for (int a = 1; a < Zi.size(); a++) {
		Eigen::VectorXd za = Eigen::VectorXd::Zero(g);
		for (int b = 0; b < Zi[a].size(); b++)
		  za += Zi[a][b];
		z += W[a] * za;
	  }
	  // Sigma_z
	  if (W[0] > 0) {
		auto temp = Zi[0][0] - z;
		Sz = W[0] * temp * temp.transpose();
	  }
	  else
		Sz = Eigen::MatrixXd::Zero(g, g);
	  for (int a = 1; a < Zi.size(); a++) {
		Eigen::MatrixXd Sz_a = Eigen::MatrixXd::Zero(g, g);
		for (int b = 0; b < Zi[a].size(); b++) {
		  Eigen::VectorXd temp = Zi[a][b] - z;
		  Sz_a += (temp * temp.transpose());
		}
		Sz += Sz_a * W[a];
	  }
	  // Sigma _zx
	  Sxz = Eigen::MatrixXd::Zero(n, g);
	  for (int a = 1; a < Zi.size(); a++) {
		Eigen::MatrixXd Sxz_a = Eigen::MatrixXd::Zero(n, g);
		for (int b = 0; b < m; b++)
		  Sxz_a += Lscaled[a - 1].col(b) * (Zi[a][b + m] - Zi[a][b]).transpose();
		Sxz += Sxz_a * W[a];
	  }
	}

	template<typename Func>
	void SelUT(const Eigen::VectorXd& x, const Eigen::MatrixXd& Sx,
		const Eigen::VectorXi& inl, Func fin, int N, Eigen::VectorXd& z,
		Eigen::MatrixXd& Sz, Eigen::MatrixXd& Sxz) {
		int n = (int)x.size();
		// nonlinear dependency vector
		Eigen::VectorXi NL = Eigen::VectorXi::Zero(n);
		for (int i = 0; i < inl.size(); i++)
			NL[inl[i]] = 1;
		// partial Choleski
		Eigen::MatrixXd L = PartialChol(Sx, NL);
		SelUTCore(x, L, fin, N, z, Sz, Sxz);
	}

	template<typename Func>
	void SelUT(const Eigen::VectorXd& x, const Eigen::MatrixXd& Sx, int m,
		Func fin, int N, Eigen::VectorXd& z, Eigen::MatrixXd& Sz, Eigen::MatrixXd& Sxz,
		Eigen::SparseMatrix<double> Q) {
		int n = (int)x.size();
		// nonlinear dependency vector
		Eigen::VectorXi NL = Eigen::VectorXi::Ones(n);
		for (int k = m; k < n; k++)
			NL[k] = 0;
		// partial Choleski
		Eigen::MatrixXd L = PartialChol(Sx, NL);
		if (Q.rows() > 0)
		  L = Q.transpose() * L;
		SelUTCore(x, L, fin, N, z, Sz, Sxz);
	}
}
