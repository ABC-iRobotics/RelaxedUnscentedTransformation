#include "INES_lib.h"
#include <iostream>

using namespace UT_INES;
using namespace Eigen;

#include <chrono>
#include <random>

const double Ts = 0.05;

Eigen::VectorXd RandomVector(const Eigen::MatrixXd& S) {
  static std::default_random_engine generator;
  static std::normal_distribution<double> distribution(0, 1);
  Eigen::VectorXd out(S.cols());
  for (int i = 0; i < S.cols(); i++)
	out(i) = sqrt(S(i, i)) * distribution(generator);
  return out;
}

Eigen::VectorXd GetInput(double t, const Eigen::MatrixXd& Sw) {
  Eigen::VectorXd out = RandomVector(Sw);
  out(0) += sin(t * 0.2);
  out(1) += cos(t * 0.5);
  return out;
}

ValWithCov InitialEstimation(const Eigen::VectorXd& xtrue) {
  Eigen::MatrixXd Sx = Eigen::MatrixXd::Zero(11, 11);
  Sx(0, 0) = 0.1;
  Sx(1, 1) = 0.001;
  Sx(2, 2) = 0.1;
  Sx(3, 3) = 0.1;
  Sx(4, 4) = 0.1;
  Sx(5, 5) = 0.1;
  Sx(6, 6) = 0.1;
  Sx(7, 7) = 0.001;
  Sx(8, 8) = 0.1;
  Sx(9, 9) = 0.1;
  Sx(10, 10) = 0.1;
  Sx = 20 * Sx;
  return ValWithCov(xtrue+RandomVector(Sx), Sx);
}

Eigen::MatrixXd getSw() {
  Eigen::MatrixXd Sw_ = Eigen::MatrixXd::Identity(8, 8) * 1;
  Sw_(0, 0) = 1.5;
  Sw_(1, 1) = 1.5;
  return Sw_;
}

Eigen::MatrixXd getSv() {
  return Eigen::MatrixXd::Identity(6, 6) * 0.5;
}

class Filter {
private:
  bool useRelaxed;
  std::vector<double> alpha;
  ValWithCov x_est;

  struct Sim {
	double RMSE = 0;
	unsigned int n = 0;
  };

  std::vector<Sim> RMSE;

public:
  struct FilterSettings {
	bool useRelaxed;
	std::vector<double> alpha;
	FilterSettings(bool useRelaxed_, std::vector<double> alpha_) :
	  useRelaxed(useRelaxed_), alpha(alpha_) {};
  };

  Filter(const FilterSettings& set) :
	useRelaxed(set.useRelaxed), alpha(set.alpha), x_est({}, {}) {}

  void Step(double dT, const Eigen::VectorXd& y_meas) {
	static auto Sw = getSw();
	static auto Sv = getSv();
	/////////////// Filtering ////////////////
	  // Prediction step
	ValWithCov y_est({}, {});
	if (useRelaxed) {
	  x_est = RelaxedUTN_INES1(dT, x_est, Sw, alpha);
	  y_est = RelaxedUTN_INES2(x_est, Sv, alpha);
	}
	else {
	  x_est = FullUTN_INES1(dT, x_est, Sw, alpha);
	  y_est = FullUTN_INES2(x_est, Sv, alpha);
	}
	/*
	std::cout << "x_true: " << x_true.transpose() << std::endl <<
	  " x_est " << x_est.y.transpose() << std::endl;
	std::cout << "S_est: " << x_est.Sy << std::endl << std::endl;

	std::cout << "y_est: " << y_est.y.transpose() << std::endl;
	std::cout << "Sy_est: " << y_est.Sy << std::endl << std::endl;*/
	// Kalman-filtering
	{
	  Eigen::MatrixXd K = y_est.Sxy * y_est.Sy.inverse();
	  Eigen::VectorXd x_est_new = x_est.y + K * (y_meas - y_est.y);
	  Eigen::MatrixXd Sx_est_new = x_est.Sy - K * y_est.Sxy.transpose();

	  x_est = { x_est_new ,Sx_est_new };
	  
	  /*
	  std::cout << "x_true: " << x_true.transpose() << std::endl <<
		" x_est " << x_est_new.transpose() << std::endl;
	  std::cout << "S_est: " << Sx_est_new << std::endl << std::endl;
	  */
	}
  }

  Eigen::VectorXd GetX() {
	return x_est.y;
  }

  void newSim(const ValWithCov& x_) {
	x_est = x_;
	RMSE.push_back({});
  }

  void computeRMSE(const Eigen::VectorXd& x_true) {
	auto n = RMSE.size() - 1;
	RMSE[n].RMSE += (x_est.y - x_true).transpose() * (x_est.y - x_true);
	RMSE[n].n++;
  }

  double getRMSE() const {
	double out = 0;
	for (auto& it : RMSE) {
	  out += sqrt(it.RMSE / double(it.n));
	}
	return out / double(RMSE.size());
  }
};

std::vector<double> RMSE(int N, int K, const std::vector<Filter::FilterSettings>& set) {
  // Initialize filters
  std::vector<Filter> filters;
  for (auto& it : set)
	filters.push_back({ it });
  // Covariance matrices
  auto Sw = getSw();
  auto Sv = getSv();
  // Value to be returned
  double dT = 0.02;
  for (int n = 0; n < N; n++) {
	/// Initial values
	Eigen::VectorXd x_true(11);
	x_true << 0.1, 0, 0., 0., 0., 0.5, 0.5, 0., 0., 0., 0.;
	// Estmated value
	auto x_est = InitialEstimation(x_true);
	// init filters
	for (auto& it : filters)
	  it.newSim(x_est);
	for (int k = 0; k < K; k++) {
	  //////////////// Simulation ////////////////
	  // Simulation: update x
	  x_true = UT_INES::StateUpdate(x_true, GetInput(double(k) * dT, Sw), dT);
	  // Simulation: compute z
	  Eigen::VectorXd y_meas = UT_INES::TrueOutput(x_true) + GetInput(double(k) * dT, Sv);
	  for (auto& it : filters) {
		it.Step(dT, y_meas);
		it.computeRMSE(x_true);
	  }
	}
  }
  std::vector<double> out;
  for (auto& it : filters)
	out.push_back(it.getRMSE());
  return out;
}

int main() {

  auto RMSE_ = RMSE(1, 10000, { {false,{1}} });

  for (auto it: RMSE_)
	std::cout << it << std::endl;

  return 0;
}