#ifndef CHOLESKY_H
#define CHOLESKY_H

#include "Eigen/Dense"

namespace UT {

  /// <summary>
  ///  Considering the matrix 'a' the method provides matrix 'out' such that out*out' is equal to 'a'
  ///  in columns/row listed in 'inl'
  /// </summary>
  /// <param name="a"> input matrix (semi-positive definite) of size nxn</param>
  /// <param name="inl"> vector of indices where the factorization must be performed of length m with values 0,..,n-1 </param>
  /// <returns> a matrix of size nxj where j<=m </returns>
  Eigen::MatrixXd PartialChol(const Eigen::MatrixXd& a, const Eigen::VectorXi& inl);

  /// <summary>
  ///  Considering the matrix 'a' the method provides matrix 'out' such that out*out' is equal to 'a'
  /// </summary>
  /// <param name="a"> input matrix (semi-positive definite) of size nxn</param>
  /// <returns> a matrix of size nxj where j<=n </returns>
  Eigen::MatrixXd FullChol(const Eigen::MatrixXd& a);

}

#endif