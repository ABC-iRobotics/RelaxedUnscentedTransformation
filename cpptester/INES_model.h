#pragma once
#include "Model.h"

namespace UT_INES {
  class Model2DAbsRel : public Model {
	Eigen::MatrixXd getA1(double Ts) const override;

	Eigen::VectorXi getil1() const override;

	Eigen::VectorXi getinl1() const override;

	Eigen::VectorXi getg1() const override;

	Eigen::MatrixXd getF1() const override;

	Eigen::VectorXd f1(const Eigen::VectorXd& x, double dT) const override;

	Eigen::VectorXd f1full(const Eigen::VectorXd& x, double dT) const override;

	//TODO: not only for wa, weps?
	Eigen::MatrixXd getB1(double dT) const override;

	Eigen::MatrixXd getA2() const override;

	Eigen::VectorXi getil2() const override;

	Eigen::VectorXi getinl2() const override;

	Eigen::VectorXi getg2() const override;

	Eigen::MatrixXd getF2() const override;

	Eigen::VectorXd f2(const Eigen::VectorXd& x) const override;

	Eigen::VectorXd f2full(const Eigen::VectorXd& x) const override;

  public:
	Eigen::VectorXd GetInput(double t) const override;

	Eigen::VectorXd InitialValue() const override;

	ValWithCov InitialEstimation() const override;

	Eigen::MatrixXd getSw() const override;

	Eigen::MatrixXd getSv() const override;
  };
}
