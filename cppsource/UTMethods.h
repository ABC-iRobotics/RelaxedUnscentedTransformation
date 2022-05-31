#pragma once
#include "UTComponents.h"
#include "UTCore.h"
#include "UTMultiscaledCore.h"
/// <summary>
///	  The header gives example howto build methods from the methods defined under the namespace of UTComponents.
///   But you can build your own method around your problem from the components.
/// </summary>

namespace UTMethods {

  /// <summary>
  ///	Considering the function y = f(x) (where f:Rn->Rf) function, E(x) and Sigma_xx,
  ///	the method determines E(y), Sigma_yy and Sigma_xy via UT.
  /// </summary>
  /// <typeparam name="Func"> Type of the f function. </typeparam>
  /// <param name="f"> An f: Rn->Rf function </param>
  /// <param name="x"> Contains E(x) vector length n and Sigma_xx matrix of size nxn </param>
  /// <returns> Contains E(y) vector of length f, Sigma_yy matrix of size fxf, Sigma_xy matrix of size nxf </returns>
  /// 
  template <typename Func>
  UT::ValWithCov FullUT(Func f, const UT::ValWithCov& x) {
	auto xdiffs = UTComponents::GenSigmaDifferencesFull(x.Sy);
	return UTComponents::UTCore(x.y, xdiffs, f, UTOriginal(xdiffs.size()));
  }

  /// <summary>
  ///	Considering the function y = f(x) (where f:Rn->Rf) function, E(x) and Sigma_xx,
  ///	the method determines E(y), Sigma_yy and Sigma_xy via MultiScaled UT.
  /// </summary>
  /// <typeparam name="Func"> Type of the f function. </typeparam>
  /// <typeparam name="SettingsGenerator"></typeparam>
  /// <param name="f"> An f: Rn->Rf function </param>
  /// <param name="x"> Contains E(x) vector length n and Sigma_xx matrix of size nxn </param>
  /// <returns> Contains E(y) vector of length f, Sigma_yy matrix of size fxf, Sigma_xy matrix of size nxf </returns>
  /// 
  template <typename Func>
  UT::ValWithCov FullUTN(Func f, const UT::ValWithCov& x) {
	auto xdiffs = UTComponents::GenSigmaDifferencesFull(x.Sy);
	return UTComponents::MultiScaledUTCore(x, xdiffs, f, UTOriginalAsMulti(xdiffs.size()));
  }

  /// <summary>
  ///	Considering the function y = A*x(il) + b(g),
  ///   where b=[b0;F*b0], b0=f(x(inl)), f:Rn->Rj, E(x) and Sigma_xx,
  ///	the method determines E(y), Sigma_yy and Sigma_xy via UT.
  /// </summary>
  /// <typeparam name="Func"> Type of the f function. </typeparam>
  /// <param name="A"> Coefficient matrix of size fxh </param>
  /// <param name="il"> Indices of x (from 0,..,n-1) that must be multiplied with matrix A of length h </param>
  /// <param name="f"> An f: Rn->Rj function </param>
  /// <param name="F"> Coefficient matrix of size kxj </param>
  /// <param name="g"> Indices for vector b (from 0,..,(k+j-1),
  ///  but can contain the same index multiple times as well) of length f </param>
  /// <param name="inl"> index vector of length >=m, on 0,..,(n-1) </param>
  /// <param name="x"> Contains E(x) vector length n and Sigma_xx matrix of size nxn </param>
  /// <returns> Contains E(y) vector of length f, Sigma_yy matrix of size fxf, Sigma_xy matrix of size nxf </returns>
  /// 
  template <typename Func>
  UT::ValWithCov RelaxedUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il, Func f,
	const Eigen::MatrixXd& F, const Eigen::VectorXi& g, const Eigen::VectorXi& inl,
	const UT::ValWithCov& x) {
	auto xdiffs = UTComponents::GenSigmaDifferences(x.Sy, inl);
	auto b0 = UTComponents::UTCore(x.y, xdiffs, f, UTOriginal(xdiffs.size()));
	auto b = UTComponents::LinearMappingOnb(b0, F);
	return UTComponents::MixedLinSourcesWithReordering(x, b, il, A, g);
  }

  /// <summary>
  ///	Considering the function y = A*x(il) + b(g),
  ///   where b=[b0;F*b0], b0=f(x(inl)), f:Rn->Rj, E(x) and Sigma_xx,
  ///	the method determines E(y), Sigma_yy and Sigma_xy via MultiScaled UT.
  /// </summary>
  /// <typeparam name="Func"> Type of the f function. </typeparam>
  /// <param name="A"> Coefficient matrix of size fxh </param>
  /// <param name="il"> Indices of x (from 0,..,n-1) that must be multiplied with matrix A of length h </param>
  /// <param name="f"> An f: Rn->Rj function </param>
  /// <param name="F"> Coefficient matrix of size kxj </param>
  /// <param name="g"> Indices for vector b (from 0,..,(k+j-1),
  ///  but can contain the same index multiple times as well) of length f </param>
  /// <param name="inl"> index vector of length >=m, on 0,..,(n-1) </param>
  /// <param name="x"> Contains E(x) vector length n and Sigma_xx matrix of size nxn </param>
  /// <returns> Contains E(y) vector of length f, Sigma_yy matrix of size fxf, Sigma_xy matrix of size nxf </returns>
  /// 
  template <typename Func>
  UT::ValWithCov RelaxedUTN(const Eigen::MatrixXd& A, const Eigen::VectorXi& il, Func f,
	const Eigen::MatrixXd& F, const Eigen::VectorXi& g, const Eigen::VectorXi& inl,
	const UT::ValWithCov& x) {
	auto xdiffs = UTComponents::GenSigmaDifferences(x.Sy, inl);
	auto b0 = UTComponents::MultiScaledUTCore(x, xdiffs, f, UTOriginalAsMulti(xdiffs.size()));
	auto b = UTComponents::LinearMappingOnb(b0, F);
	return UTComponents::MixedLinSourcesWithReordering(x, b, il, A, g);
  }
}