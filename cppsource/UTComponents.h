#ifndef UTCOMPONENT_H
#define UTCOMPONENT_H

#include <vector>
#include <Eigen/SparseCore>
#include "Stoch.h"

namespace UT {

  /// <summary>
  /// The function generates std::vector of delta_1,...,delta_m vectors such that
  ///     S = 1/(2n) * [  sum_(i=1,..,m) delta_i * delta_i^T + sum_(i=1,..,m) (-delta_i) * (-delta_i)^T ]
  /// via Cholesky-factorization to generate sigma points as
  ///     X0 = E(x); Xi = E(x) - alpha*delta_i; X(i+m) = E(x) + alpha*delta_i
  /// where m=n if S is full-rank
  /// </summary>
  /// <param name="S"> covariance (symmetric semi-positive definite) matrix of size n x n </param>
  /// <returns> std::vector of delta_1,...,delta_n vectors of length n </returns>
  ///
  std::vector<Eigen::VectorXd> GenSigmaDifferencesFull(const Eigen::MatrixXd& S);

  /// <summary>
  /// The function generates std::vector of delta_1,...,delta_m vectors such that the sum of
  ///     1/(2n) * [  sum_(i=1,..,m) delta_i * delta_i^T + sum_(i=1,..,m) (-delta_i) * (-delta_i)^T ]
  /// describes well the columns and rows of matrix S with indices provided in vector inl.
  /// Provided via partial Cholesky-factorization to generate sigma points as
  ///     X0 = E(x); Xi = E(x) - alpha*delta_i; X(i+m) = E(x) + alpha*delta_i
  /// </summary>
  /// <param name="S"> covariance (symmetric semi-positive definite) matrix of size n x n </param>
  /// <param name="inl"> index vector of length >=m, on 0,..,(n-1) </param>
  /// <returns> std::vector of delta_1,...,delta_m vectors of length n </returns>
  /// 
  std::vector<Eigen::VectorXd> GenSigmaDifferences(const Eigen::MatrixXd& S, const Eigen::VectorXi& inl);

  

  /// <summary>
  ///	Describes the y = Q*x transformation (Q is an ortohogonal matrix), where the f function will depend on the first m values.
  ///	Its constructor implements a practical method to initialize these values.
  /// </summary>
  struct ExactSubspace {
	/*! \brief Type to describe if the function depends on the M*x(i) combination of the values of x with indices i */
	struct MixedNonlin {
	  Eigen::VectorXi i;
	  Eigen::VectorXd M;
	  MixedNonlin(Eigen::VectorXi i, Eigen::VectorXd M) : i(i), M(M) {};
	};
	/*! \brief Type to describe a list of such a dependencies*/
	typedef std::vector<MixedNonlin> MixedNonlinearityList;

	Eigen::SparseMatrix<double> Q;
	int m;

	/// <summary>
	///	  Returns the first m columns of matrix Q. The function depends nonlinearly on values Q1*x.
	/// </summary>
	/// <returns> The first m columns of matrix Q </returns>
	const Eigen::Block<const Eigen::SparseMatrix<double>,-1,-1,false> Q1() const {
	  return Q.block(0, 0, m, Q.cols());
	}

	/// <summary>
	///	  The constructor implements a method to obtain the important quantities of the subspace.
	///	  If the function depends on variables of x with indices given in vector inl and
	///	  on combinations of variables with indices ik and weights wk, and there are combinations
	///	  like this given as {{i1,w1},{i2,w2},...}
	/// </summary>
	/// <param name="n"> number of variables </param>
	/// <param name="inl"> indices of single variables, the function depends on nonlinearly </param>
	/// <param name="mix"> combinations of variables the function depend on nonlinearly with
	///  indices ik and weights wk, and there are combinations
	///	  like this given as {{i1,w1},{i2,w2},...} </param>
	/// 
	ExactSubspace(int n, const Eigen::VectorXi& inl, const MixedNonlinearityList& mix);

	/// <summary>
	/// Dummy constructor
	/// </summary>
	ExactSubspace() {};
  };
  
  /// <summary>
  /// The function generates std::vector of delta_1,...,delta_m vectors such that the sum of
  ///     1/(2n) * [  sum_(i=1,..,m) delta_i * delta_i^T + sum_(i=1,..,m) (-delta_i) * (-delta_i)^T ]
  /// describes well the part of matrix S that will be affect the nonlinear function considered (and described by struct "sp").
  /// Provided via partial Cholesky-factorization to generate sigma points as
  ///     X0 = E(x); Xi = E(x) - alpha*delta_i; X(i+m) = E(x) + alpha*delta_i
  /// </summary>
  /// <param name="S"> covariance (symmetric semi-positive definite) matrix of size n x n </param>
  /// <param name="sp"> descriptor of the subspace of nonlinearity of type ExactSubspace </param>
  /// <returns> std::vector of delta_1,...,delta_m vectors of length n </returns>
  /// 
  std::vector<Eigen::VectorXd> GenSigmaDifferencesFromExactSubspace(const Eigen::MatrixXd& S,
	const ExactSubspace& sp);

  /// <summary>
  ///	Considering E(b0), Sigma_b0b0, Sigma_xb0 and function b = [b0; F * b0], the method determines E(b), Sigma_bb, Sigma_xb
  /// </summary>
  /// <param name="b0"> Contains E(b0) vector length l, Sigma_b0b0 matrix of size lxl, Sigma_xb0 matrix of size nxl </param>
  /// <param name="F"> Coefficient matrix of size kxl </param>
  /// <returns> Contains E(b) vector of length (k+l), Sigma_bb matrix of size (k+l)x(k+l), Sigma_xb matrix of size nx(k+l) </returns>
  /// 
  UT::ValWithCov LinearMappingOnb(const UT::ValWithCov& b0, const Eigen::MatrixXd& F);

  /// <summary>
  ///	Considering E(b0), Sigma_b0b0, Sigma_xb0 and function b = [0;b0;F * b0], the method determines E(b), Sigma_bb, Sigma_xb
  /// </summary>
  /// <param name="b0"> Contains E(b0) vector length l, Sigma_b0b0 matrix of size lxl, Sigma_xb0 matrix of size nxl </param>
  /// <param name="F"> Coefficient matrix of size kxl </param>
  /// <returns> Contains E(b) vector of length j, Sigma_bb matrix of size jxj, Sigma_xb matrix of size nxj, where j=k+l+1 </returns>
  /// 
  UT::ValWithCov LinearMappingOnbWith0(const UT::ValWithCov& b0, const Eigen::MatrixXd& F);

  /// <summary>
  ///	Considering E(x), Sigma_xx, E(b), Sigma_bb, Sigma_xb and function y = A*x(il) + b(g),
  ///	the method determines E(y), Sigma_yy and Sigma_xy 
  /// </summary>
  /// <param name="x"> Contains E(x) vector length n and Sigma_xx matrix of size nxn </param>
  /// <param name="b"> Contains E(b) vector of length j, Sigma_bb matrix of size jxj, Sigma_xb matrix of size nxj </param>
  /// <param name="il"> Indices of x (from 0,..,n-1) that must be multiplied with matrix A of length h </param>
  /// <param name="A"> Coefficient matrix of size fxh</param>
  /// <param name="g"> Indices for vector b (from 0,..,(j-1),
  ///  but can contain the same index multiple times as well) of length f </param>
  /// <returns> Contains E(y) vector of length f, Sigma_yy matrix of size fxf, Sigma_xy matrix of size nxf </returns>
  /// 
  UT::ValWithCov MixedLinSourcesWithReordering(const UT::ValWithCov& x, const UT::ValWithCov& b,
	const Eigen::VectorXi& il, const Eigen::MatrixXd& A, const Eigen::VectorXi& g);

  /// <summary>
  ///	Considering E(x), Sigma_xx, E(b), Sigma_bb, Sigma_xb and function y = A*x(il) + b,
  ///	the method determines E(y), Sigma_yy and Sigma_xy 
  /// </summary>
  /// <param name="x"> Contains E(x) vector length n and Sigma_xx matrix of size nxn </param>
  /// <param name="b"> Contains E(b) vector of length f, Sigma_bb matrix of size fxf, Sigma_xb matrix of size nxf </param>
  /// <param name="il"> Indices of x (from 0,..,n-1) that must be multiplied with matrix A of length h </param>
  /// <param name="A"> Coefficient matrix of size fxh</param>
  /// <returns> Contains E(y) vector of length f, Sigma_yy matrix of size fxf, Sigma_xy matrix of size nxf </returns>
  /// 
  UT::ValWithCov MixedLinSources(const UT::ValWithCov& x, const UT::ValWithCov& b,
	const Eigen::VectorXi& il, const Eigen::MatrixXd& A);
}

#endif
