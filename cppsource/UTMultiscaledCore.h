#ifndef UTMULTISCALEDCORE_H
#define UTMULTISCALEDCORE_H

#include <vector>
#include "Stoch.h"

namespace UT {

  /// <summary>
  /// Struct to describe the settings to be applied in the Multiscaled UT
  /// </summary>
  struct MultiScaledUTSettings {
	std::vector<double> kappa, W, V;
	double W0, V0;
	
	int Nmax() const { return kappa.size(); }

	MultiScaledUTSettings(const std::vector<double>& kappa_, const std::vector<double>& W_,
	  const std::vector<double>& V_, double W0_, double V0_) : kappa(kappa_), W(W_), V(V_), W0(W0_), V0(V0_) {}

	MultiScaledUTSettings(const std::vector<double>& kappa_, const std::vector<double>& W_,
	  double W0_) : kappa(kappa_), W(W_), V(W_), W0(W0_), V0(W0_) {}
  };

  MultiScaledUTSettings UTOriginalAsMulti(int n) {
	return MultiScaledUTSettings({ double(n) }, { 0.5 / double(n) }, 0.);
  }

  MultiScaledUTSettings UTKappa3AsMulti(int n) {
	return MultiScaledUTSettings({ 3. }, { 0.5 / 3. }, 1. - double(n) / 3.);
  }

  MultiScaledUTSettings UTN2(int n, double alpha0, double alpha1) {
	double k0 = 3. + sqrt(6 * alpha0 / alpha1), k1 = 3. - sqrt(6 * alpha1 / alpha0);
	return MultiScaledUTSettings({ k0, k1 },
	  { 0.5 / k0, 0.5 / k1 }, { 0.5 / k0, 0.5 / k1 }, 1. - double(n) * (1. / k0 + 1. / k1), 2. - double(n) * (1. / k0 + 1. / k1));
  }

  /// <summary>
  ///	Considering E(x) of length n, std::vector of delta_1,...,delta_m vectors of length n, and an f: Rn->Rj function,
  ///	the method computes X0 = E(x), XiN = E(x) - sqrt(kappaN)* delta_i, X(i+m)N = E(x) + sqrt(kappaN)* delta_i sigma points,
  ///	map them as Y0=f(X0), YiN = f(XiN), and computes
  ///	E(y) ~ W0*Y0 + W1*sum_(i=1)^(2m) Yi1 + W2*sum_(i=1)^(2m) Yi2;
  ///	Sigma_yy ~ V0*(Y0-E(y))*(Y0-E(y))' + V1 * sum_(i=1)^(2m)(Yi1-E(y))*(Yi1-E(y))'
  ///		 + V2 * sum_(i=1)^(2m)(Yi2-E(y))*(Yi2-E(y))'
  ///	Sigma_xy ~ sqrt(V1*W1) * sum_(i=1)^(2m)(Xi1-E(x))*(Yi1-E(y))' + sqrt(V2*W2) * sum_(i=1)^(2m)(Xi2-E(x))*(Yi2-E(y))'
  /// </summary>
  /// <typeparam name="Func"> Type of the f function. </typeparam>
  /// <param name="x"> The value of E(x) of length n </param>
  /// <param name="xdiffs"> std::vector of delta_1,...,delta_m vectors of length n </param>
  /// <param name="f"> An f: Rn->Rj function </param>
  /// <param name="settings"> Contains W0,W1,W2,V0,V1,V2 and kappa1,kappa2 values to be used. </param>
  /// <returns> Contains E(y) vector of length j, Sigma_yy matrix of size jxj, Sigma_xy matrix of size nxj </returns>
  ///
  template<typename Func>
  UT::ValWithCov MultiScaledUTCore(const Eigen::VectorXd& x, const std::vector<Eigen::VectorXd>& xdiffs, Func f,
	const MultiScaledUTSettings& settings) {
	int n = (int)x.size();
	int Nmax = settings.Nmax();
	int m = xdiffs.size();
	// Scale offsets
	std::vector<std::vector<Eigen::VectorXd>> scaledxdiffs;
	for (int N = 0; N < Nmax; N++) {
	  scaledxdiffs.push_back({});
	  for (int i = 0; i < m; i++)
		scaledxdiffs[N].push_back(xdiffs[i] * sqrt(settings.kappa[N]));// *settings.U[N][n]);
	}
	// Map the sigma points
	Eigen::VectorXd Z0 = f(x);
	std::vector<std::vector<Eigen::VectorXd>> Zi;
	for (int N = 0; N < Nmax; N++) {
	  Zi.push_back({});
	  for (int b = 0; b < m; b++)
		Zi[N].push_back(f(x - scaledxdiffs[N][b]));
	  for (int b = 0; b < m; b++)
		Zi[N].push_back(f(x + scaledxdiffs[N][b]));
	}
	// Expected z value
	int g = (int)Z0.size();
	Eigen::VectorXd z = settings.W0 * Z0;
	for (int a = 0; a < Zi.size(); a++) {
	  Eigen::VectorXd za = Eigen::VectorXd::Zero(g);
	  for (int b = 0; b < Zi[a].size(); b++)
		za += Zi[a][b];
	  z += settings.W[a] * za;
	}
	// Sigma_z
	Eigen::MatrixXd Sz;
	{
	  Eigen::VectorXd temp = Z0 - z;
	  Sz = settings.V0 * temp * temp.transpose()*0;
	}
	for (int a = 1; a < Zi.size(); a++) {
	  Eigen::MatrixXd Sz_a = Eigen::MatrixXd::Zero(g, g);
	  for (int b = 0; b < Zi[a].size(); b++) {
		Eigen::VectorXd temp = Zi[a][b] - z;
		Sz_a += (temp * temp.transpose());
	  }
	  Sz += Sz_a * settings.V[a];
	}
	// Sigma _zx
	Eigen::MatrixXd Sxz = Eigen::MatrixXd::Zero(n, g);
	for (int a = 0; a < Zi.size(); a++) {
	  Eigen::MatrixXd Sxz_a = Eigen::MatrixXd::Zero(n, g);
	  for (int b = 0; b < m; b++)
		Sxz_a += scaledxdiffs[a][b] * (Zi[a][b + m] - Zi[a][b]).transpose();
	  Sxz += Sxz_a * sqrt(settings.V[a]* settings.W[a]);
	}
	return UT::ValWithCov(z, Sz, Sxz);
  }
}

#endif