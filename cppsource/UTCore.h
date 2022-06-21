#ifndef UTCORE_H
#define UTCORE_H

#include <vector>
#include "Stoch.h"

namespace UT {

  /// <summary>
  /// Struct to describe the settings to be applied in the UT
  /// </summary>
  struct UTSettings {
	double kappa;
	double W0, W1;
	double V0, V1;
	UTSettings(double kappa_, double W0_, double W1_, double V0_, double V1_) :
	  kappa(kappa_), W0(W0_), W1(W1_), V0(V0_), V1(V1_) {}
	UTSettings(double kappa_, double W0_, double W1_) :
	  kappa(kappa_), W0(W0_), W1(W1_), V0(W0_), V1(W1_) {}
  };

  /// <summary>
  /// Function that initializes a UTSettings from the number of n (number of sigma points are 2n+1) as
  /// kappa = n, W1 = 1/2n, W0 = 0, V1 = W1, V0 = W0
  /// </summary>
  /// <param name="n"> the number of sigma points are 2n+1 </param>
  /// <returns> UTSettings parameters </returns>
  inline UTSettings UTOriginal(int n) { return UTSettings(n, 0, 0.5 / double(n)); };

  /// <summary>
  /// Function that initializes a UTSettings from the number of n (number of sigma points are 2n+1)  as
  /// kappa = 3, W1 = 1/2kappa, W0 = 1-2nW1, V1 = W1, V0 = W0
  /// </summary>
  /// <param name="n"> the number of sigma points are 2n+1 </param>
  /// <returns> UTSettings parameters </returns>
  inline UTSettings UTKappa3(int n) { return UTSettings(3, 1. - double(n)/3., 0.5 / 3.); };

  /// <summary>
  ///	Considering E(x) of length n, std::vector of delta_1,...,delta_m vectors of length n, and an f: Rn->Rj function,
  ///	the method computes X0 = E(x), Xi = E(x) - sqrt(kappa)* delta_i, X(i+m) = E(x) + sqrt(kappa)* delta_i sigma points,
  ///	map them as Yi = f(Xi), and computes
  ///	E(y) ~ W0*Y0 + W1*sum_(i=1)^(2m) Yi;
  ///	Sigma_yy ~ V0*(Y0-E(y))*(Y0-E(y))' + V1 * sum_(i=1)^(2m)(Yi-E(y))*(Yi-E(y))'
  ///	Sigma_xy ~ sqrt(V1*W1) * sum_(i=1)^(2m)(Xi-E(x))*(Yi-E(y))'
  /// </summary>
  /// <typeparam name="Func"> Type of the f function. </typeparam>
  /// <param name="x"> The value of E(x) of length n </param>
  /// <param name="xdiffs"> std::vector of delta_1,...,delta_m vectors of length n </param>
  /// <param name="f"> An f: Rn->Rj function </param>
  /// <param name="settings"> Contains W0,W1,V0,V1 and kappa values to be used. </param>
  /// <returns> Contains E(y) vector of length j, Sigma_yy matrix of size jxj, Sigma_xy matrix of size nxj </returns>
  ///
  template<typename Func>
  UT::ValWithCov UTCore(const Eigen::VectorXd& x, const std::vector<Eigen::VectorXd>& xdiffs, Func f,
	const UTSettings& settings) {
	int n = (int)x.size();
	int m = xdiffs.size();
	// Scale offsets
	std::vector<Eigen::VectorXd> scaledxdiffs;
	double sqrtkappa = sqrt(settings.kappa);
	for (int i = 0; i < m; i++)
	  scaledxdiffs.push_back(xdiffs[i] * sqrtkappa);
	// Map the sigma points
	Eigen::VectorXd Z0 = f(x);
	std::vector<Eigen::VectorXd> Zi;
	for (int b = 0; b < m; b++)
	  Zi.push_back(f(x - scaledxdiffs[b]));
	for (int b = 0; b < m; b++)
	  Zi.push_back(f(x + scaledxdiffs[b]));
	// Expected z value
	int g = (int)Z0.size();
	Eigen::VectorXd z = settings.W0 * Z0;
	{
	  Eigen::VectorXd za = Eigen::VectorXd::Zero(g);
	  for (int b = 0; b < Zi.size(); b++)
		za += Zi[b];
	  z += settings.W1 * za;
	}
	// Sigma_z
	Eigen::MatrixXd Sz;
	{
	  auto temp = Z0 - z;
	  Sz = settings.V0 * temp * temp.transpose();
	}
	{
	  Eigen::MatrixXd Sz_a = Eigen::MatrixXd::Zero(g, g);
	  for (int b = 0; b < Zi.size(); b++) {
		Eigen::VectorXd temp = Zi[b] - z;
		Sz_a += (temp * temp.transpose());
	  }
	  Sz += Sz_a * settings.V1;
	}
	// Sigma _zx
	Eigen::MatrixXd Sxz = Eigen::MatrixXd::Zero(n, g);
	{
	  Eigen::MatrixXd Sxz_a = Eigen::MatrixXd::Zero(n, g);
	  for (int b = 0; b < m; b++)
		Sxz_a += scaledxdiffs[b] * (Zi[b + m] - Zi[b]).transpose();
	  Sxz += Sxz_a * settings.V1;
	}
	return UT::ValWithCov(z, Sz, Sxz);
  }
}

#endif