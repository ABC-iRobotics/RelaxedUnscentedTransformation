#pragma once
#include "UTComponents.h"

namespace UT_INES {
  class Model {
	// To specify state update
	virtual Eigen::MatrixXd getA1(double Ts) const = 0;

	virtual Eigen::VectorXi getil1() const = 0;

	virtual Eigen::VectorXi getinl1() const = 0;

	virtual Eigen::VectorXi getg1() const = 0;

	virtual Eigen::MatrixXd getF1() const = 0;

	virtual Eigen::VectorXd f1(const Eigen::VectorXd& x, double dT) const = 0;

	virtual Eigen::VectorXd f1full(const Eigen::VectorXd& x, double dT) const = 0;

	// To specify output update
	virtual Eigen::MatrixXd getB1(double Ts) const = 0;

	virtual Eigen::MatrixXd getA2() const = 0;

	virtual Eigen::VectorXi getil2() const = 0;

	virtual Eigen::VectorXi getinl2() const = 0;

	virtual Eigen::VectorXi getg2() const = 0;

	virtual Eigen::MatrixXd getF2() const = 0;

	virtual Eigen::VectorXd f2(const Eigen::VectorXd& x) const = 0;

	virtual Eigen::VectorXd f2full(const Eigen::VectorXd& x) const = 0;

  public:
	virtual Eigen::VectorXd GetInput(double t) const = 0;

	virtual Eigen::VectorXd InitialValue() const = 0;

	virtual ValWithCov InitialEstimation() const = 0;

	virtual Eigen::MatrixXd getSw() const = 0;

	virtual Eigen::MatrixXd getSv() const = 0;

	virtual Eigen::VectorXd StateUpdate(const Eigen::VectorXd& x, const Eigen::VectorXd& w, double dT) const;

	virtual Eigen::VectorXd TrueOutput(const Eigen::VectorXd& x) const;

	virtual Eigen::VectorXd StateUpdate2(const Eigen::VectorXd& x, const Eigen::VectorXd& w, double dT) const;

	virtual Eigen::VectorXd TrueOutput2(const Eigen::VectorXd& x) const;

	virtual ValWithCov RelaxedUT_INES1(double dT, const ValWithCov& x0_in,
	  const Eigen::MatrixXd& Sw) const;

	virtual ValWithCov RelaxedUTN_INES1(double dT, const ValWithCov& x0,
	  const Eigen::MatrixXd& Sw, const std::vector<double>& alpha) const;

	virtual ValWithCov FullUT_INES1(double dT, const ValWithCov& x0,
	  const Eigen::MatrixXd& Sw) const;

	virtual ValWithCov FullUTN_INES1(double dT, const ValWithCov& x0,
	  const Eigen::MatrixXd& Sw, const std::vector<double>& alpha) const;

	virtual ValWithCov RelaxedUT_INES2(const ValWithCov& x0,
	  const Eigen::MatrixXd& Sv) const;

	virtual ValWithCov RelaxedUTN_INES2(const ValWithCov& x0,
	  const Eigen::MatrixXd& Sv, const std::vector<double>& alpha) const;

	virtual ValWithCov FullUT_INES2(const ValWithCov& x0,
	  const Eigen::MatrixXd& Sv) const;

	virtual ValWithCov FullUTN_INES2(const ValWithCov& x0,
	  const Eigen::MatrixXd& Sv, const std::vector<double>& alpha) const;

	ValWithCov UT_INES1(double dT, const ValWithCov& x0,
	  const Eigen::MatrixXd& Sw, const std::vector<double>& alpha, bool useRelaxed, bool useHO) const;

	ValWithCov UT_INES2(const ValWithCov& x0, const Eigen::MatrixXd& Sv,
	  const std::vector<double>& alpha, bool useRelaxed, bool useHO) const;
  };
}
