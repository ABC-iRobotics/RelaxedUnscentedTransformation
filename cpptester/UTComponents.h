#pragma once
#include <vector>
#include "Eigen/Dense"

struct MultiScaledUTSettings {
  int Nmax;
  std::vector<double> kappa;
  std::vector<double> alpha;
  //std::vector<std::vector<Eigen::MatrixXd>> U;
  /*
  void initU(std::vector<double> phi, int nMax) {
	for (int N = 0; N < Norder - 1; N++) {
	  std::vector<Eigen::MatrixXd> temp;
	  for (int n = 0; n < nMax; n++) {
		Eigen::MatrixXd out = Eigen::MatrixXd::Identity(n, n);
		double c = cos(phi[N]), s = sin(phi[N]);
		for (int i = 0; i < n - 1; i++) {
		  Eigen::MatrixXd temp = Eigen::MatrixXd::Identity(n, n);
		  temp(i, i) = c;
		  temp(i, i + 1) = -s;
		  temp(i + 1, i) = s;
		  temp(i + 1, i + 1) = c;
		  out = out * temp;
		}
		temp.push_back(out);
	  }
	  U.push_back(temp);
	}
  }*/

  MultiScaledUTSettings(int n) {
	kappa.push_back(3);
	alpha.push_back(1);
	Nmax = 1;
  };

  MultiScaledUTSettings(int n,std::vector<double> alpha_) {
	alpha = alpha_;
	Nmax = alpha_.size();

	if (Nmax == 1)
	  kappa.push_back(3); 
	if (Nmax == 2) {
	  kappa.push_back(3. + sqrt(6 * alpha[0] / alpha[1]));
	  kappa.push_back(3. - sqrt(6 * alpha[1] / alpha[0]));
	}
  }
};

struct UTSettings {
  double kappa, alpha;
  double W0, W1;

  UTSettings(int n) {
	kappa = 3; alpha = 1;
	W1 = 0.5 / kappa;
	W0 = 1. - n / kappa;
  };
};

struct ValWithCov {
  Eigen::VectorXd y;
  Eigen::MatrixXd Sy, Sxy;
  ValWithCov(const Eigen::VectorXd y_, const Eigen::MatrixXd& Sy_,
	const Eigen::MatrixXd& Sxy_) : y(y_), Sy(Sy_), Sxy(Sxy_) {};
  ValWithCov(const Eigen::VectorXd y_, const Eigen::MatrixXd& Sy_) : y(y_), Sy(Sy_) {};
};

Eigen::VectorXd RandomVector(const Eigen::MatrixXd& S);

namespace UTComponents {

  Eigen::MatrixXd PartialChol(const Eigen::MatrixXd& a, const Eigen::VectorXi& inl);

  Eigen::MatrixXd FullChol(const Eigen::MatrixXd& a);

  std::vector<Eigen::VectorXd> GenSigmaDifferences(const Eigen::MatrixXd& S, const Eigen::VectorXi& inl);

  std::vector<Eigen::VectorXd> GenSigmaDifferencesFull(const Eigen::MatrixXd& S);

  template<typename Func>
  ValWithCov UTCore(const Eigen::VectorXd& x, const std::vector<Eigen::VectorXd>& xdiffs, Func fin,
	const UTSettings& settings) {
	int n = (int)x.size();
	int m = xdiffs.size();
	// Scale offsets
	std::vector<Eigen::VectorXd> scaledxdiffs;
	for (int i = 0; i < m; i++)
	  scaledxdiffs.push_back(xdiffs[i] * sqrt(settings.kappa));
	// Map the sigma points
	Eigen::VectorXd Z0 = fin(x);
	std::vector<Eigen::VectorXd> Zi;
	for (int b = 0; b < m; b++)
	  Zi.push_back(fin(x - scaledxdiffs[b]));
	for (int b = 0; b < m; b++)
	  Zi.push_back(fin(x + scaledxdiffs[b]));
	// Compute weights
	double W1 = 0.5 / settings.kappa / settings.alpha;
	double V1 = W1;
	double W0 = 1. - W1 * 2. * double(m);
	double V0 = W0 + 3 - settings.kappa / m;
	// Expected z value
	int g = (int)Z0.size();
	Eigen::VectorXd z = W0 * Z0;
	{
	  Eigen::VectorXd za = Eigen::VectorXd::Zero(g);
	  for (int b = 0; b < Zi.size(); b++)
		za += Zi[b];
	  z += W1 * za;
	}
	// Sigma_z
	Eigen::MatrixXd Sz;
	{
	  auto temp = Z0 - z;
	  Sz = V0 * temp * temp.transpose();
	}
	{
	  Eigen::MatrixXd Sz_a = Eigen::MatrixXd::Zero(g, g);
	  for (int b = 0; b < Zi.size(); b++) {
		Eigen::VectorXd temp = Zi[b] - z;
		Sz_a += (temp * temp.transpose());
	  }
	  Sz += Sz_a * V1;
	}
	// Sigma _zx
	Eigen::MatrixXd Sxz = Eigen::MatrixXd::Zero(n, g);
	{
	  Eigen::MatrixXd Sxz_a = Eigen::MatrixXd::Zero(n, g);
	  for (int b = 0; b < m; b++)
		Sxz_a += scaledxdiffs[b] * (Zi[b + m] - Zi[b]).transpose();
	  Sxz += Sxz_a * V1;
	}
	return ValWithCov(z, Sz, Sxz);
  }

  template<typename Func>
  ValWithCov MultiScaledUTCore(const Eigen::VectorXd& x, const std::vector<Eigen::VectorXd>& xdiffs, Func fin,
	const MultiScaledUTSettings& settings) {
	int n = (int)x.size();
	int Nmax = settings.Nmax;
	int m = xdiffs.size();
	// Scale offsets
	std::vector<std::vector<Eigen::VectorXd>> scaledxdiffs;
	for (int N = 0; N < Nmax; N++) {
	  scaledxdiffs.push_back({});
	  for (int i = 0; i < m; i++)
		scaledxdiffs[N].push_back(xdiffs[i] * sqrt(settings.kappa[N]));// *settings.U[N][n]);
	}
	// Map the sigma points
	std::vector<std::vector<Eigen::VectorXd>> Zi;
	Zi.push_back({ fin(x) });
	for (int N = 0; N < Nmax; N++) {
	  Zi.push_back({});
	  for (int b = 0; b < m; b++)
		Zi[N + 1].push_back(fin(x - scaledxdiffs[N][b]));
	  for (int b = 0; b < m; b++)
		Zi[N + 1].push_back(fin(x + scaledxdiffs[N][b]));
	}
	// Compute weights
	Eigen::VectorXd W(Nmax + 1);
	W[0] = 0;
	for (int l = 1; l < Nmax + 1; l++) // W0 will be computed later
	  W[l] = 0.5 / double(Nmax) / settings.kappa[l - 1] / settings.alpha[l - 1];
	W[0] = 1. - W.sum() * 2. * double(m);
	auto V = W;
	double kappa_sum = 0;
	for (auto it : settings.kappa)
	  kappa_sum += it;
	V[0] += 3 - kappa_sum / m;
	// Expected z value
	int g = (int)Zi[0][0].size();
	Eigen::VectorXd z = W[0] * Zi[0][0];
	for (int a = 1; a < Zi.size(); a++) {
	  Eigen::VectorXd za = Eigen::VectorXd::Zero(g);
	  for (int b = 0; b < Zi[a].size(); b++)
		za += Zi[a][b];
	  z += W[a] * za;
	}
	// Sigma_z
	Eigen::MatrixXd Sz;
	{
	  Eigen::VectorXd temp = Zi[0][0] - z;
	  Sz = V[0] * temp * temp.transpose()*0;
	}
	for (int a = 1; a < Zi.size(); a++) {
	  Eigen::MatrixXd Sz_a = Eigen::MatrixXd::Zero(g, g);
	  for (int b = 0; b < Zi[a].size(); b++) {
		Eigen::VectorXd temp = Zi[a][b] - z;
		Sz_a += (temp * temp.transpose());
	  }
	  Sz += Sz_a * V[a];
	}
	// Sigma _zx
	Eigen::MatrixXd Sxz = Eigen::MatrixXd::Zero(n, g);
	for (int a = 1; a < Zi.size(); a++) {
	  Eigen::MatrixXd Sxz_a = Eigen::MatrixXd::Zero(n, g);
	  for (int b = 0; b < m; b++)
		Sxz_a += scaledxdiffs[a - 1][b] * (Zi[a][b + m] - Zi[a][b]).transpose();
	  Sxz += Sxz_a * V[a];
	}
	return ValWithCov(z, Sz, Sxz);
  }

  ValWithCov LinearMappingOnb(const ValWithCov& b0_in, const Eigen::MatrixXd& F);

  ValWithCov LinearMappingOnbWith0(const ValWithCov& b0_in, const Eigen::MatrixXd& F);

  ValWithCov MixedLinSourcesWithReordering(const ValWithCov& x0_in, const ValWithCov& b_in,
	const Eigen::VectorXi& il, const Eigen::MatrixXd& A, const Eigen::VectorXi& g);

  template <typename Func>
  ValWithCov RelaxedUT(const Eigen::MatrixXd& A, const Eigen::VectorXi& il, Func fin, int N,
	const Eigen::MatrixXd& F, const Eigen::VectorXi& g, const Eigen::VectorXi& inl, const ValWithCov& x0_in) {
	auto xdiffs = UTComponents::GenSigmaDifferences(x0_in.Sy, inl);
	auto b0 = UTComponents::UTCore(x0_in.y, xdiffs, fin, {});
	auto b = UTComponents::LinearMappingOnb(b0, F);
	return UTComponents::MixedLinSourcesWithReordering(x0_in, b, il, A, g);
  }

  template <typename Func>
  ValWithCov RelaxedUTN(const Eigen::MatrixXd& A, const Eigen::VectorXi& il, Func fin, int N,
	const Eigen::MatrixXd& F, const Eigen::VectorXi& g, const Eigen::VectorXi& inl, const ValWithCov& x0_in) {
	auto xdiffs = UTComponents::GenSigmaDifferences(x0_in.Sy, inl);
	auto b0 = UTComponents::MultiScaledUTCore(x0, xdiffs, fin, {});
	auto b = UTComponents::LinearMappingOnb(b0, F);
	return UTComponents::MixedLinSourcesWithReordering(x0_in, b, il, A, g);
  }

  template <typename Func>
  ValWithCov FullUT(Func fin, const ValWithCov& x0) {
	auto xdiffs = UTComponents::GenSigmaDifferencesFull(x0_in.Sy);
	return = UTComponents::UTCore(x0_in.y, xdiffs, fin, {});
  }

  template <typename Func>
  ValWithCov FullUTN(Func fin, const ValWithCov& x0) {
	auto xdiffs = UTComponents::GenSigmaDifferencesFull(x0_in.Sy);
	return UTComponents::MultiScaledUTCore(x0, xdiffs, fin, {});
  }
}