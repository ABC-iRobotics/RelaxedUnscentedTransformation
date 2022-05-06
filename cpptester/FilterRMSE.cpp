#include "Model.h"
#include "index_selector.h"
#include "FilterRMSE.h"
using namespace UT_INES;

void UT_INES::Filter::Step(double dT, const Eigen::VectorXd& y_meas, const Eigen::MatrixXd& Sw, const Eigen::MatrixXd& Sv) {
  /////////////// Filtering ////////////////
  // Prediction step
  ValWithCov y_est({}, {});
  x_est = model->UT_StateUpdate(dT, x_est, Sw, settings.alpha, settings.useRelaxed, settings.useHO);
  y_est = model->UT_OutputUpdate(x_est, Sv, settings.alpha, settings.useRelaxed, settings.useHO);
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

Eigen::VectorXd UT_INES::Filter::GetX() const {
  return x_est.y;
}

void UT_INES::Filter::newSim(const ValWithCov& x_) {
  x_est = x_;
  RMSE.push_back({});
}

void UT_INES::Filter::computeRMSE(const Eigen::VectorXd& x_true) {
  auto n = RMSE.size() - 1;
  RMSE[n].RMSE += (x_est.y - x_true).transpose() * (x_est.y - x_true);
  RMSE[n].n++;
}

double UT_INES::Filter::getRMSE() const {
  double out = 0;
  for (auto& it : RMSE)
	out += sqrt(it.RMSE / double(it.n));
  return out / double(RMSE.size());
}

std::vector<double> UT_INES::RMSE(double dT, int N, int K, const std::vector<Filter::FilterSettings>& set, Model* model) {
  // Initialize filters
  std::vector<Filter> filters;
  for (auto& it : set)
	filters.push_back({ it ,model });
  // Covariance matrices
  auto Sw = model->getSw();
  auto Sv = model->getSv();
  // Value to be returned
  for (int n = 0; n < N; n++) {
	/// Initial values
	Eigen::VectorXd x_true = model->InitialValue();
	// Estmated value
	auto x_est = model->InitialEstimation();
	// init filters
	for (auto& it : filters)
	  it.newSim(x_est);
	for (int k = 0; k < K; k++) {
	  //////////////// Simulation ////////////////
	  // Simulation: update x
	  x_true = model->StateUpdate_Full(x_true, model->GetInput(double(k) * dT), dT);
	  // Simulation: compute z
	  Eigen::VectorXd y_meas = model->TrueOutput_Full(x_true) + RandomVector(Sv);
	  for (auto& it : filters) {
		it.Step(dT, y_meas, Sw, Sv);
		it.computeRMSE(x_true);
	  }
	}
  }
  std::vector<double> out;
  for (auto& it : filters)
	out.push_back(it.getRMSE());
  return out;
}

UT_INES::Filter::FilterSettings::FilterSettings(bool useRelaxed_, bool useHO_, std::vector<double> alpha_) :
  useRelaxed(useRelaxed_), alpha(alpha_), useHO(useHO_) {}


Filter::Filter(const FilterSettings& set, Model* model_) :
  settings(set), x_est({}, {}), model(model_) {}