#pragma once
#include "Model.h"

//#define useHO true

namespace UT_INES {

  class Filter {
  public:
	struct FilterSettings {
	  bool useRelaxed, useHO;
	  std::vector<double> alpha;
	  FilterSettings(bool useRelaxed_, bool useHO, std::vector<double> alpha_);
	};

  private:
	FilterSettings settings;
	ValWithCov x_est;

	struct Sim {
	  double RMSE = 0;
	  unsigned int n = 0;
	};

	std::vector<Sim> RMSE;
	Model* model;

  public:
	Filter(const FilterSettings& set, Model* model_);

	void Step(double dT, const Eigen::VectorXd& y_meas,
	  const Eigen::MatrixXd& Sw, const Eigen::MatrixXd& Sv);

	Eigen::VectorXd GetX() const;

	void newSim(const ValWithCov& x_);

	void computeRMSE(const Eigen::VectorXd& x_true);

	double getRMSE() const;
  };

  std::vector<double> RMSE(double dT, int N, int K, const std::vector<Filter::FilterSettings>& set, Model* model);
}
