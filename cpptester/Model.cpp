#include "Model.h"
#include "index_selector.h"
using namespace UT_INES;

ValWithCov Model::UT_StateUpdate(double dT, const ValWithCov& x0,
  const Eigen::MatrixXd& Sw, const std::vector<double>& alpha, bool useRelaxed, bool useHO) const {
  if (useRelaxed && useHO)
	return RelaxedUTN_StateUpdate(dT, x0, Sw, alpha);
  if (useRelaxed && !useHO)
	return RelaxedUT_StateUpdate(dT, x0, Sw);
  if (!useRelaxed && useHO)
	return FullUTN_StateUpdate(dT, x0, Sw, alpha);
  if (!useRelaxed && !useHO)
	return FullUT_StateUpdate(dT, x0, Sw);
}

ValWithCov Model::UT_OutputUpdate(const ValWithCov& x0,
  const Eigen::MatrixXd& Sv, const std::vector<double>& alpha, bool useRelaxed, bool useHO) const {
  if (useRelaxed && useHO)
	return RelaxedUTN_OutputUpdate(x0, Sv, alpha);
  if (useRelaxed && !useHO)
	return RelaxedUT_OutputUpdate(x0, Sv);
  if (!useRelaxed && useHO)
	return FullUTN_OutputUpdate(x0, Sv, alpha);
  if (!useRelaxed && !useHO)
	return FullUT_OutputUpdate(x0, Sv);
}

Eigen::VectorXd UT_INES::Model::StateUpdate_Full(const Eigen::VectorXd& x,
  const Eigen::VectorXd& w, double dT) const {
  return ffull_SU(x, dT) + getB_SU(dT) * w;
}

Eigen::VectorXd UT_INES::Model::TrueOutput_Full(const Eigen::VectorXd& x) const {
  return ffull_OU(x);
}

Eigen::VectorXd UT_INES::Model::StateUpdate_Sep(const Eigen::VectorXd& x,
  const Eigen::VectorXd& w, double dT) const {
  Eigen::VectorXd b0 = f_SU(x, dT);
  Eigen::VectorXd b(b0.size() + 1);
  b.segment(1, b0.size()) = b0;
  b(0) = 0;
  b = RelaxedUnscentedTransformation::VectorSelect(b, getg_SU());
  return b + getA_SU(dT) * RelaxedUnscentedTransformation::VectorSelect(x, getil_SU()) + getB_SU(dT) * w;
}

Eigen::VectorXd UT_INES::Model::TrueOutput_Sep(const Eigen::VectorXd& x) const {
  Eigen::VectorXd b0 = f_OU(x);
  Eigen::VectorXd b(b0.size() + 1);
  b.segment(1, b0.size()) = b0;
  b(0) = 0;
  b = RelaxedUnscentedTransformation::VectorSelect(b, getg_OU());
  return b + getA_OU() * RelaxedUnscentedTransformation::VectorSelect(x, getil_OU());
}

ValWithCov UT_INES::Model::RelaxedUT_StateUpdate(double dT,
  const ValWithCov& x0, const Eigen::MatrixXd& Sw) const {
  static Eigen::VectorXi inl = getinl_SU();
  auto xdiffs = UTComponents::GenSigmaDifferences(x0.Sy, inl);
  auto b0 = UTComponents::UTCore(x0.y, xdiffs,
	[dT, this](const Eigen::VectorXd& in)->Eigen::VectorXd {return f_SU(in, dT); }, { (int)xdiffs.size() });
  auto b = UTComponents::LinearMappingOnbWith0(b0, getF_SU());
  static Eigen::VectorXi il = getil_SU();
  static Eigen::VectorXi g = getg_SU();
  auto out = UTComponents::MixedLinSourcesWithReordering(x0, b, il, getA_SU(dT), g);
  static Eigen::MatrixXd B = getB_SU(dT);
  out.Sy += B * Sw * B.transpose();
  return out;
}

ValWithCov UT_INES::Model::RelaxedUTN_StateUpdate(double dT, const ValWithCov& x0,
  const Eigen::MatrixXd& Sw, const std::vector<double>& alpha) const {
  static Eigen::VectorXi inl = getinl_SU();
  auto xdiffs = UTComponents::GenSigmaDifferences(x0.Sy, inl);
  auto b0 = UTComponents::MultiScaledUTCore(x0.y, xdiffs,
	[dT, this](const Eigen::VectorXd& in)->Eigen::VectorXd {return f_SU(in, dT); }, { (int)xdiffs.size(),{ 1.,1. } });
  auto b = UTComponents::LinearMappingOnbWith0(b0, getF_SU());
  static Eigen::VectorXi il = getil_SU();
  static Eigen::VectorXi g = getg_SU();
  auto out = UTComponents::MixedLinSourcesWithReordering(x0, b, il, getA_SU(dT), g);
  static Eigen::MatrixXd B = getB_SU(dT);
  out.Sy += B * Sw * B.transpose();
  return out;
}

ValWithCov UT_INES::Model::FullUT_StateUpdate(double dT, const ValWithCov& x0,
  const Eigen::MatrixXd& Sw) const {
  auto xdiffs = UTComponents::GenSigmaDifferencesFull(x0.Sy);
  auto out = UTComponents::UTCore(x0.y, xdiffs,
	[dT, this](const Eigen::VectorXd& in)->Eigen::VectorXd {return ffull_SU(in, dT); }, { (int)xdiffs.size() });
  static Eigen::MatrixXd B = getB_SU(dT);
  out.Sy += B * Sw * B.transpose();
  return out;
}

ValWithCov UT_INES::Model::FullUTN_StateUpdate(double dT, const ValWithCov& x0,
  const Eigen::MatrixXd& Sw, const std::vector<double>& alpha) const {
  auto xdiffs = UTComponents::GenSigmaDifferencesFull(x0.Sy);
  auto out = UTComponents::MultiScaledUTCore(x0.y, xdiffs,
	[dT, this](const Eigen::VectorXd& in)->Eigen::VectorXd {return ffull_SU(in, dT); }, { (int)xdiffs.size(), alpha });
  static Eigen::MatrixXd B = getB_SU(dT);
  out.Sy += B * Sw * B.transpose();
  return out;
}

ValWithCov UT_INES::Model::RelaxedUT_OutputUpdate(const ValWithCov& x0,
  const Eigen::MatrixXd& Sv) const {
  static Eigen::VectorXi inl = getinl_OU();
  auto xdiffs = UTComponents::GenSigmaDifferences(x0.Sy, inl);
  auto b0 = UTComponents::UTCore(x0.y, xdiffs,
	[this](const Eigen::VectorXd& x)->Eigen::VectorXd { return f_OU(x); }, { (int)x0.y.size() });
  auto b = UTComponents::LinearMappingOnbWith0(b0, getF_OU());
  static Eigen::VectorXi il = getil_OU();
  static Eigen::VectorXi g = getg_OU();
  auto out = UTComponents::MixedLinSourcesWithReordering(x0, b, il, getA_OU(), g);
  out.Sy += Sv;
  return out;
}

ValWithCov UT_INES::Model::RelaxedUTN_OutputUpdate(const ValWithCov& x0,
  const Eigen::MatrixXd& Sv, const std::vector<double>& alpha) const {
  static Eigen::VectorXi inl = getinl_OU();
  auto xdiffs = UTComponents::GenSigmaDifferences(x0.Sy, inl);
  auto b0 = UTComponents::MultiScaledUTCore(x0.y, xdiffs,
	[this](const Eigen::VectorXd& x)->Eigen::VectorXd { return f_OU(x); }, { (int)xdiffs.size(), alpha });
  auto b = UTComponents::LinearMappingOnbWith0(b0, getF_OU());
  static Eigen::VectorXi il = getil_OU();
  static Eigen::VectorXi g = getg_OU();
  auto out = UTComponents::MixedLinSourcesWithReordering(x0, b, il, getA_OU(), g);
  out.Sy += Sv;
  return out;
}

ValWithCov UT_INES::Model::FullUT_OutputUpdate(const ValWithCov& x0,
  const Eigen::MatrixXd& Sv) const {
  auto xdiffs = UTComponents::GenSigmaDifferencesFull(x0.Sy);
  auto out = UTComponents::UTCore(x0.y, xdiffs,
	[this](const Eigen::VectorXd& x)->Eigen::VectorXd { return ffull_OU(x); }, { (int)xdiffs.size() });
  out.Sy += Sv;
  return out;
}

ValWithCov UT_INES::Model::FullUTN_OutputUpdate(const ValWithCov& x0,
  const Eigen::MatrixXd& Sv, const std::vector<double>& alpha) const {
  auto xdiffs = UTComponents::GenSigmaDifferencesFull(x0.Sy);
  auto out = UTComponents::MultiScaledUTCore(x0.y, xdiffs,
	[this](const Eigen::VectorXd& x)->Eigen::VectorXd { return ffull_OU(x); }, { (int)xdiffs.size(), alpha });
  out.Sy += Sv;
  return out;
}
