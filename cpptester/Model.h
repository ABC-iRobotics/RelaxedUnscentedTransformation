#pragma once
#include "UTComponents.h"

namespace UT_INES {
  class Model {
	// To specify state update
	virtual Eigen::MatrixXd getA_SU(double Ts) const = 0;

	virtual Eigen::VectorXi getil_SU() const = 0;

	virtual Eigen::VectorXi getinl_SU() const = 0;

	virtual Eigen::VectorXi getg_SU() const = 0;

	virtual Eigen::MatrixXd getF_SU() const = 0;

	virtual Eigen::VectorXd f_SU(const Eigen::VectorXd& x, double dT) const = 0;

	virtual Eigen::VectorXd ffull_SU(const Eigen::VectorXd& x, double dT) const = 0;

	// To specify output update
	virtual Eigen::MatrixXd getB_SU(double Ts) const = 0;

	virtual Eigen::MatrixXd getA_OU() const = 0;

	virtual Eigen::VectorXi getil_OU() const = 0;

	virtual Eigen::VectorXi getinl_OU() const = 0;

	virtual Eigen::VectorXi getg_OU() const = 0;

	virtual Eigen::MatrixXd getF_OU() const = 0;

	virtual Eigen::VectorXd f_OU(const Eigen::VectorXd& x) const = 0;

	virtual Eigen::VectorXd ffull_OU(const Eigen::VectorXd& x) const = 0;

  public:
	virtual Eigen::VectorXd GetInput(double t) const = 0;

	virtual Eigen::VectorXd InitialValue() const = 0;

	virtual ValWithCov InitialEstimation() const = 0;

	virtual Eigen::MatrixXd getSw() const = 0;

	virtual Eigen::MatrixXd getSv() const = 0;

	virtual Eigen::VectorXd StateUpdate_Full(const Eigen::VectorXd& x, const Eigen::VectorXd& w, double dT) const;

	virtual Eigen::VectorXd TrueOutput_Full(const Eigen::VectorXd& x) const;

	virtual Eigen::VectorXd StateUpdate_Sep(const Eigen::VectorXd& x, const Eigen::VectorXd& w, double dT) const;

	virtual Eigen::VectorXd TrueOutput_Sep(const Eigen::VectorXd& x) const;

	virtual ValWithCov RelaxedUT_StateUpdate(double dT, const ValWithCov& x0_in,
	  const Eigen::MatrixXd& Sw) const;

	virtual ValWithCov RelaxedUTN_StateUpdate(double dT, const ValWithCov& x0,
	  const Eigen::MatrixXd& Sw, const std::vector<double>& alpha) const;

	virtual ValWithCov FullUT_StateUpdate(double dT, const ValWithCov& x0,
	  const Eigen::MatrixXd& Sw) const;

	virtual ValWithCov FullUTN_StateUpdate(double dT, const ValWithCov& x0,
	  const Eigen::MatrixXd& Sw, const std::vector<double>& alpha) const;

	virtual ValWithCov RelaxedUT_OutputUpdate(const ValWithCov& x0,
	  const Eigen::MatrixXd& Sv) const;

	virtual ValWithCov RelaxedUTN_OutputUpdate(const ValWithCov& x0,
	  const Eigen::MatrixXd& Sv, const std::vector<double>& alpha) const;

	virtual ValWithCov FullUT_OutputUpdate(const ValWithCov& x0,
	  const Eigen::MatrixXd& Sv) const;

	virtual ValWithCov FullUTN_OutputUpdate(const ValWithCov& x0,
	  const Eigen::MatrixXd& Sv, const std::vector<double>& alpha) const;

	ValWithCov UT_StateUpdate(double dT, const ValWithCov& x0,
	  const Eigen::MatrixXd& Sw, const std::vector<double>& alpha, bool useRelaxed, bool useHO) const;

	ValWithCov UT_OutputUpdate(const ValWithCov& x0, const Eigen::MatrixXd& Sv,
	  const std::vector<double>& alpha, bool useRelaxed, bool useHO) const;
  };
}
