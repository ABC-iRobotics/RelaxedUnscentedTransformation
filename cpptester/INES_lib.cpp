#include "INES_lib.h"
#include "index_selector.h"

using namespace UT_INES;

Eigen::MatrixXd getA1(double Ts) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Identity(11, 11);
  A(0, 7) = Ts;
  return A;
}

Eigen::VectorXi getil1() {
  Eigen::VectorXi il(11);
  for (int i = 0; i < 11; i++)
	il[i] = i;
  return il;
}

Eigen::VectorXi getinl1() {
  Eigen::VectorXi inl = Eigen::VectorXi::Zero(3);
  inl(0) = 0;
  inl(1) = 1;
  inl(2) = 4;
  return inl;
}

Eigen::VectorXi getg1() {
  Eigen::VectorXi g = Eigen::VectorXi::Zero(11);
  g(5) = 1;
  g(6) = 2;
  g(9) = 3;
  g(10) = 4;
  return g;
}

Eigen::MatrixXd getF1() {
  return Eigen::MatrixXd(0, 4);
}

Eigen::VectorXd f1(const Eigen::VectorXd& x, double dT) {
  double phi = x(0);
  double Dphi = x(4);
  double v = x(1);
  Eigen::VectorXd out(4);
  out(0) = dT * v * cos(phi);
  out(1) = dT * v * sin(phi);
  out(2) = dT * v * (cos(phi + Dphi) - cos(phi));
  out(3) = dT * v * (sin(phi + Dphi) + sin(phi));
  return out;
}

Eigen::VectorXd f1full(const Eigen::VectorXd& x, double dT) {
  double phi = x(0);
  double v = x(1);
  double Dphi = x(4);
  Eigen::VectorXd out = x;
  out(0) += dT * x(7);
  out(5) += dT * v * cos(phi);
  out(6) += dT * v * sin(phi);
  out(9) += dT * v * (cos(phi + Dphi) - cos(phi));
  out(10) += dT * v * (sin(phi + Dphi) + sin(phi));
  return out;
}

Eigen::MatrixXd getB1() {
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(11, 8);
  B(1, 0) = 1;
  B(2, 2) = 1;
  B(3, 3) = 1;
  B(4, 7) = 1;
  B(7, 1) = 1;
  B(8, 4) = 1;
  B(9, 5) = 1;
  B(10, 6) = 1;
  return B;
}

Eigen::MatrixXd getA2() {
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6, 8);
  A(0, 2) = 1;
  A(1, 3) = 1;
  A(2, 0) = 1;
  A(2, 5) = 1;
  A(3, 2) = 1;
  A(3, 6) = 1;
  A(4, 3) = 1;
  A(4, 7) = 1;
  A(5, 0) = 1;
  A(5, 1) = 1;
  return A;
}

Eigen::VectorXi getil2() {
  Eigen::VectorXi il(8);
  il(0) = 0;
  il(1) = 4;
  il(2) = 5;
  il(3) = 6;
  il(4) = 7;
  il(5) = 8;
  il(6) = 9;
  il(7) = 10;
  return il;
}

Eigen::VectorXi getinl2() {
  Eigen::VectorXi inl = Eigen::VectorXi::Zero(3);
  inl(0) = 0;
  inl(1) = 2;
  inl(2) = 3;
  return inl;
}

Eigen::VectorXi getg2() {
  Eigen::VectorXi g = Eigen::VectorXi::Zero(6);
  g(0) = 1;
  g(1) = 2;
  return g;
}

Eigen::MatrixXd getF2() {
  return Eigen::MatrixXd(0, 2);
}

Eigen::VectorXd f2(const Eigen::VectorXd& x) {
  double phi = x(0);
  double xc = x(2);
  double yc = x(3);
  Eigen::VectorXd out(2);
  out(0) = xc * cos(phi) - yc * sin(phi);
  out(1) = yc * cos(phi) + xc * sin(phi);
  return out;
}

Eigen::VectorXd f2full(const Eigen::VectorXd& x) {
  double phi = x(0);
  double xc = x(2);
  double yc = x(3);
  Eigen::VectorXd out(6);
  out(0) = x(5) + xc * cos(phi) - yc * sin(phi);
  out(1) = x(6) + yc * cos(phi) + xc * sin(phi);
  out(2) = x(0) + x(8);
  out(3) = x(5) + x(9);
  out(4) = x(6) + x(10);
  out(5) = x(0) + x(4);
  return out;
}

Eigen::VectorXd UT_INES::StateUpdate(const Eigen::VectorXd& x, const Eigen::VectorXd& w, double dT) {
  return f1full(x,dT) + dT*getB1()*w;
}

Eigen::VectorXd UT_INES::TrueOutput(const Eigen::VectorXd& x) {
  return f2full(x);
}

Eigen::VectorXd UT_INES::StateUpdate2(const Eigen::VectorXd& x, const Eigen::VectorXd& w, double dT) {
  Eigen::VectorXd b0 = f1(x, dT);
  Eigen::VectorXd b(b0.size() + 1);
  b.segment(1, b0.size()) = b0;
  b(0) = 0;
  b = RelaxedUnscentedTransformation::VectorSelect(b, getg1());
  return b + getA1(dT)* RelaxedUnscentedTransformation::VectorSelect(x, getil1()) + dT * getB1() * w;
}

Eigen::VectorXd UT_INES::TrueOutput2(const Eigen::VectorXd& x) {
  Eigen::VectorXd b0 = f2(x);
  Eigen::VectorXd b(b0.size() + 1);
  b.segment(1, b0.size()) = b0;
  b(0) = 0;
  b = RelaxedUnscentedTransformation::VectorSelect(b, getg2());
  return b + getA2() * RelaxedUnscentedTransformation::VectorSelect(x, getil2());
}

ValWithCov UT_INES::RelaxedUT_INES1(double dT, const ValWithCov& x0, const Eigen::MatrixXd& Sw) {
  static Eigen::VectorXi inl = getinl1();
  auto xdiffs = UTComponents::GenSigmaDifferences(x0.Sy, inl);
  auto b0 = UTComponents::UTCore(x0.y, xdiffs,
	[dT](const Eigen::VectorXd& in)->Eigen::VectorXd {return f1(in, dT); }, { (int)xdiffs.size() });
  auto b = UTComponents::LinearMappingOnbWith0(b0, getF1());
  static Eigen::VectorXi il = getil1();
  static Eigen::VectorXi g = getg1();
  auto out = UTComponents::MixedLinSourcesWithReordering(x0, b, il, getA1(dT), g);
  static Eigen::MatrixXd B = getB1();
  out.Sy += dT*dT*B * Sw * B.transpose();
  return out;
}

ValWithCov UT_INES::RelaxedUTN_INES1(double dT, const ValWithCov& x0,
  const Eigen::MatrixXd& Sw, const std::vector<double>& alpha) {
  static Eigen::VectorXi inl = getinl1();
  auto xdiffs = UTComponents::GenSigmaDifferences(x0.Sy, inl);
  auto b0 = UTComponents::MultiScaledUTCore(x0.y, xdiffs,
	[dT](const Eigen::VectorXd& in)->Eigen::VectorXd {return f1(in, dT); }, {(int)xdiffs.size(), {1.,1.} });
  auto b = UTComponents::LinearMappingOnbWith0(b0, getF1());
  static Eigen::VectorXi il = getil1();
  static Eigen::VectorXi g = getg1();
  auto out = UTComponents::MixedLinSourcesWithReordering(x0, b, il, getA1(dT), g);
  static Eigen::MatrixXd B = getB1();
  out.Sy += dT * dT * B * Sw * B.transpose();
  return out;
}

ValWithCov UT_INES::FullUT_INES1(double dT, const ValWithCov& x0,
  const Eigen::MatrixXd& Sw) {
  auto xdiffs = UTComponents::GenSigmaDifferencesFull(x0.Sy);
  auto out = UTComponents::UTCore(x0.y, xdiffs,
	[dT](const Eigen::VectorXd& in)->Eigen::VectorXd {return f1full(in, dT); }, { (int)xdiffs.size()});
  static Eigen::MatrixXd B = getB1();
  out.Sy += dT * dT * B * Sw * B.transpose();
  return out;
}

ValWithCov UT_INES::FullUTN_INES1(double dT, const ValWithCov& x0,
  const Eigen::MatrixXd& Sw, const std::vector<double>& alpha) {
  auto xdiffs = UTComponents::GenSigmaDifferencesFull(x0.Sy);
  auto out = UTComponents::MultiScaledUTCore(x0.y, xdiffs,
	[dT](const Eigen::VectorXd& in)->Eigen::VectorXd {return f1full(in, dT); }, { (int)xdiffs.size(), alpha });
  static Eigen::MatrixXd B = getB1();
  out.Sy += dT * dT * B * Sw * B.transpose();
  return out;
}

ValWithCov UT_INES::RelaxedUT_INES2(const ValWithCov& x0,
  const Eigen::MatrixXd& Sv) {
  static Eigen::VectorXi inl = getinl2();
  auto xdiffs = UTComponents::GenSigmaDifferences(x0.Sy, inl);
  auto b0 = UTComponents::UTCore(x0.y, xdiffs, f2, { (int)x0.y.size() });
  auto b = UTComponents::LinearMappingOnbWith0(b0, getF2());
  static Eigen::VectorXi il = getil2();
  static Eigen::VectorXi g = getg2();
  auto out = UTComponents::MixedLinSourcesWithReordering(x0, b, il, getA2(), g);
  out.Sy += Sv;
  return out;
}

ValWithCov UT_INES::RelaxedUTN_INES2(const ValWithCov& x0,
  const Eigen::MatrixXd& Sv, const std::vector<double>& alpha) {
  static Eigen::VectorXi inl = getinl2();
  auto xdiffs = UTComponents::GenSigmaDifferences(x0.Sy, inl);
  auto b0 = UTComponents::MultiScaledUTCore(x0.y, xdiffs, f2, { (int)xdiffs.size(), alpha });
  auto b = UTComponents::LinearMappingOnbWith0(b0, getF2());
  static Eigen::VectorXi il = getil2();
  static Eigen::VectorXi g = getg2();
  auto out = UTComponents::MixedLinSourcesWithReordering(x0, b, il, getA2(), g);
  out.Sy += Sv;
  return out;
}

ValWithCov UT_INES::FullUT_INES2(const ValWithCov& x0,
  const Eigen::MatrixXd& Sv) {
  auto xdiffs = UTComponents::GenSigmaDifferencesFull(x0.Sy);
  auto out = UTComponents::UTCore(x0.y, xdiffs, f2full, { (int)xdiffs.size() });
  out.Sy += Sv;
  return out;
}

ValWithCov UT_INES::FullUTN_INES2(const ValWithCov& x0,
  const Eigen::MatrixXd& Sv, const std::vector<double>& alpha) {
  auto xdiffs = UTComponents::GenSigmaDifferencesFull(x0.Sy);
  auto out = UTComponents::MultiScaledUTCore(x0.y, xdiffs, f2full, { (int)xdiffs.size(), alpha });
  out.Sy += Sv;
  return out;
}
