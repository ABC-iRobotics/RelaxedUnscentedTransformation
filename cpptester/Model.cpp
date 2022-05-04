#include "Model.h"
#include "index_selector.h"
using namespace UT_INES;

ValWithCov Model::UT_INES1(double dT, const ValWithCov& x0,
  const Eigen::MatrixXd& Sw, const std::vector<double>& alpha, bool useRelaxed, bool useHO) const {
  if (useRelaxed && useHO)
	return RelaxedUTN_INES1(dT, x0, Sw, alpha);
  if (useRelaxed && !useHO)
	return RelaxedUT_INES1(dT, x0, Sw);
  if (!useRelaxed && useHO)
	return FullUTN_INES1(dT, x0, Sw, alpha);
  if (!useRelaxed && !useHO)
	return FullUT_INES1(dT, x0, Sw);
}

ValWithCov Model::UT_INES2(const ValWithCov& x0,
  const Eigen::MatrixXd& Sv, const std::vector<double>& alpha, bool useRelaxed, bool useHO) const {
  if (useRelaxed && useHO)
	return RelaxedUTN_INES2(x0, Sv, alpha);
  if (useRelaxed && !useHO)
	return RelaxedUT_INES2(x0, Sv);
  if (!useRelaxed && useHO)
	return FullUTN_INES2(x0, Sv, alpha);
  if (!useRelaxed && !useHO)
	return FullUT_INES2(x0, Sv);
}

Eigen::VectorXd UT_INES::Model::StateUpdate(const Eigen::VectorXd& x,
  const Eigen::VectorXd& w, double dT) const {
  return f1full(x, dT) + getB1(dT) * w;
}

Eigen::VectorXd UT_INES::Model::TrueOutput(const Eigen::VectorXd& x) const {
  return f2full(x);
}

Eigen::VectorXd UT_INES::Model::StateUpdate2(const Eigen::VectorXd& x,
  const Eigen::VectorXd& w, double dT) const {
  Eigen::VectorXd b0 = f1(x, dT);
  Eigen::VectorXd b(b0.size() + 1);
  b.segment(1, b0.size()) = b0;
  b(0) = 0;
  b = RelaxedUnscentedTransformation::VectorSelect(b, getg1());
  return b + getA1(dT) * RelaxedUnscentedTransformation::VectorSelect(x, getil1()) + getB1(dT) * w;
}

Eigen::VectorXd UT_INES::Model::TrueOutput2(const Eigen::VectorXd& x) const {
  Eigen::VectorXd b0 = f2(x);
  Eigen::VectorXd b(b0.size() + 1);
  b.segment(1, b0.size()) = b0;
  b(0) = 0;
  b = RelaxedUnscentedTransformation::VectorSelect(b, getg2());
  return b + getA2() * RelaxedUnscentedTransformation::VectorSelect(x, getil2());
}

ValWithCov UT_INES::Model::RelaxedUT_INES1(double dT,
  const ValWithCov& x0, const Eigen::MatrixXd& Sw) const {
  static Eigen::VectorXi inl = getinl1();
  auto xdiffs = UTComponents::GenSigmaDifferences(x0.Sy, inl);
  auto b0 = UTComponents::UTCore(x0.y, xdiffs,
	[dT, this](const Eigen::VectorXd& in)->Eigen::VectorXd {return f1(in, dT); }, { (int)xdiffs.size() });
  auto b = UTComponents::LinearMappingOnbWith0(b0, getF1());
  static Eigen::VectorXi il = getil1();
  static Eigen::VectorXi g = getg1();
  auto out = UTComponents::MixedLinSourcesWithReordering(x0, b, il, getA1(dT), g);
  static Eigen::MatrixXd B = getB1(dT);
  out.Sy += B * Sw * B.transpose();
  return out;
}

ValWithCov UT_INES::Model::RelaxedUTN_INES1(double dT, const ValWithCov& x0,
  const Eigen::MatrixXd& Sw, const std::vector<double>& alpha) const {
  static Eigen::VectorXi inl = getinl1();
  auto xdiffs = UTComponents::GenSigmaDifferences(x0.Sy, inl);
  auto b0 = UTComponents::MultiScaledUTCore(x0.y, xdiffs,
	[dT, this](const Eigen::VectorXd& in)->Eigen::VectorXd {return f1(in, dT); }, { (int)xdiffs.size(),{ 1.,1. } });
  auto b = UTComponents::LinearMappingOnbWith0(b0, getF1());
  static Eigen::VectorXi il = getil1();
  static Eigen::VectorXi g = getg1();
  auto out = UTComponents::MixedLinSourcesWithReordering(x0, b, il, getA1(dT), g);
  static Eigen::MatrixXd B = getB1(dT);
  out.Sy += B * Sw * B.transpose();
  return out;
}

ValWithCov UT_INES::Model::FullUT_INES1(double dT, const ValWithCov& x0,
  const Eigen::MatrixXd& Sw) const {
  auto xdiffs = UTComponents::GenSigmaDifferencesFull(x0.Sy);
  auto out = UTComponents::UTCore(x0.y, xdiffs,
	[dT, this](const Eigen::VectorXd& in)->Eigen::VectorXd {return f1full(in, dT); }, { (int)xdiffs.size() });
  static Eigen::MatrixXd B = getB1(dT);
  out.Sy += B * Sw * B.transpose();
  return out;
}

ValWithCov UT_INES::Model::FullUTN_INES1(double dT, const ValWithCov& x0,
  const Eigen::MatrixXd& Sw, const std::vector<double>& alpha) const {
  auto xdiffs = UTComponents::GenSigmaDifferencesFull(x0.Sy);
  auto out = UTComponents::MultiScaledUTCore(x0.y, xdiffs,
	[dT, this](const Eigen::VectorXd& in)->Eigen::VectorXd {return f1full(in, dT); }, { (int)xdiffs.size(), alpha });
  static Eigen::MatrixXd B = getB1(dT);
  out.Sy += B * Sw * B.transpose();
  return out;
}

ValWithCov UT_INES::Model::RelaxedUT_INES2(const ValWithCov& x0,
  const Eigen::MatrixXd& Sv) const {
  static Eigen::VectorXi inl = getinl2();
  auto xdiffs = UTComponents::GenSigmaDifferences(x0.Sy, inl);
  auto b0 = UTComponents::UTCore(x0.y, xdiffs,
	[this](const Eigen::VectorXd& x)->Eigen::VectorXd { return f2(x); }, { (int)x0.y.size() });
  auto b = UTComponents::LinearMappingOnbWith0(b0, getF2());
  static Eigen::VectorXi il = getil2();
  static Eigen::VectorXi g = getg2();
  auto out = UTComponents::MixedLinSourcesWithReordering(x0, b, il, getA2(), g);
  out.Sy += Sv;
  return out;
}

ValWithCov UT_INES::Model::RelaxedUTN_INES2(const ValWithCov& x0,
  const Eigen::MatrixXd& Sv, const std::vector<double>& alpha) const {
  static Eigen::VectorXi inl = getinl2();
  auto xdiffs = UTComponents::GenSigmaDifferences(x0.Sy, inl);
  auto b0 = UTComponents::MultiScaledUTCore(x0.y, xdiffs,
	[this](const Eigen::VectorXd& x)->Eigen::VectorXd { return f2(x); }, { (int)xdiffs.size(), alpha });
  auto b = UTComponents::LinearMappingOnbWith0(b0, getF2());
  static Eigen::VectorXi il = getil2();
  static Eigen::VectorXi g = getg2();
  auto out = UTComponents::MixedLinSourcesWithReordering(x0, b, il, getA2(), g);
  out.Sy += Sv;
  return out;
}

ValWithCov UT_INES::Model::FullUT_INES2(const ValWithCov& x0,
  const Eigen::MatrixXd& Sv) const {
  auto xdiffs = UTComponents::GenSigmaDifferencesFull(x0.Sy);
  auto out = UTComponents::UTCore(x0.y, xdiffs,
	[this](const Eigen::VectorXd& x)->Eigen::VectorXd { return f2full(x); }, { (int)xdiffs.size() });
  out.Sy += Sv;
  return out;
}

ValWithCov UT_INES::Model::FullUTN_INES2(const ValWithCov& x0,
  const Eigen::MatrixXd& Sv, const std::vector<double>& alpha) const {
  auto xdiffs = UTComponents::GenSigmaDifferencesFull(x0.Sy);
  auto out = UTComponents::MultiScaledUTCore(x0.y, xdiffs,
	[this](const Eigen::VectorXd& x)->Eigen::VectorXd { return f2full(x); }, { (int)xdiffs.size(), alpha });
  out.Sy += Sv;
  return out;
}
