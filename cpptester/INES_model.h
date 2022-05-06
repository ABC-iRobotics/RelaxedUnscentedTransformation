#pragma once
#include "Model.h"

namespace UT_INES {
  class Model2DAbsRel : public Model {
	Eigen::MatrixXd getA_SU(double Ts) const override;

	Eigen::VectorXi getil_SU() const override;

	Eigen::VectorXi getinl_SU() const override;

	Eigen::VectorXi getg_SU() const override;

	Eigen::MatrixXd getF_SU() const override;

	Eigen::VectorXd f_SU(const Eigen::VectorXd& x, double dT) const override;

	Eigen::VectorXd ffull_SU(const Eigen::VectorXd& x, double dT) const override;

	//TODO: not only for wa, weps?
	Eigen::MatrixXd getB_SU(double dT) const override;

	Eigen::MatrixXd getA_OU() const override;

	Eigen::VectorXi getil_OU() const override;

	Eigen::VectorXi getinl_OU() const override;

	Eigen::VectorXi getg_OU() const override;

	Eigen::MatrixXd getF_OU() const override;

	Eigen::VectorXd f_OU(const Eigen::VectorXd& x) const override;

	Eigen::VectorXd ffull_OU(const Eigen::VectorXd& x) const override;

  public:
	Eigen::VectorXd GetInput(double t) const override;

	Eigen::VectorXd InitialValue() const override;

	ValWithCov InitialEstimation() const override;

	Eigen::MatrixXd getSw() const override;

	Eigen::MatrixXd getSv() const override;
  };
}
