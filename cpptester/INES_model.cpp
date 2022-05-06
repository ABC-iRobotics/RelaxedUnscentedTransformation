#include "INES_model.h"
#include "UTComponents.h"

using namespace UT_INES;

Eigen::MatrixXd UT_INES::Model2DAbsRel::getA_SU(double Ts) const {
  Eigen::MatrixXd A = Eigen::MatrixXd::Identity(11, 11);
  A(0, 7) = Ts;
  return A;
}

Eigen::VectorXi UT_INES::Model2DAbsRel::getil_SU() const {
  Eigen::VectorXi il(11);
  for (int i = 0; i < 11; i++)
	il[i] = i;
  return il;
}

Eigen::VectorXi UT_INES::Model2DAbsRel::getinl_SU() const {
  Eigen::VectorXi inl = Eigen::VectorXi::Zero(3);
  inl(0) = 0;
  inl(1) = 1;
  inl(2) = 4;
  return inl;
}

Eigen::VectorXi UT_INES::Model2DAbsRel::getg_SU() const {
  Eigen::VectorXi g = Eigen::VectorXi::Zero(11);
  g(5) = 1;
  g(6) = 2;
  g(9) = 3;
  g(10) = 4;
  return g;
}

Eigen::MatrixXd UT_INES::Model2DAbsRel::getF_SU() const {
  return Eigen::MatrixXd(0, 4);
}

Eigen::VectorXd UT_INES::Model2DAbsRel::f_SU(const Eigen::VectorXd& x, double dT) const {
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

Eigen::VectorXd UT_INES::Model2DAbsRel::ffull_SU(const Eigen::VectorXd& x, double dT) const {
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

//TODO: not only for wa, weps?

Eigen::MatrixXd UT_INES::Model2DAbsRel::getB_SU(double dT) const {
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(11, 8);
  B(1, 0) = dT;
  B(2, 2) = dT;
  B(3, 3) = dT;
  B(4, 7) = dT;
  B(7, 1) = dT;
  B(8, 4) = dT;
  B(9, 5) = dT;
  B(10, 6) = dT;
  return B;
}

Eigen::MatrixXd UT_INES::Model2DAbsRel::getA_OU() const {
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

Eigen::VectorXi UT_INES::Model2DAbsRel::getil_OU() const {
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

Eigen::VectorXi UT_INES::Model2DAbsRel::getinl_OU() const {
  Eigen::VectorXi inl = Eigen::VectorXi::Zero(3);
  inl(0) = 0;
  inl(1) = 2;
  inl(2) = 3;
  return inl;
}

Eigen::VectorXi UT_INES::Model2DAbsRel::getg_OU() const {
  Eigen::VectorXi g = Eigen::VectorXi::Zero(6);
  g(0) = 1;
  g(1) = 2;
  return g;
}

Eigen::MatrixXd UT_INES::Model2DAbsRel::getF_OU() const {
  return Eigen::MatrixXd(0, 2);
}

Eigen::VectorXd UT_INES::Model2DAbsRel::f_OU(const Eigen::VectorXd& x) const {
  double phi = x(0);
  double xc = x(2);
  double yc = x(3);
  Eigen::VectorXd out(2);
  out(0) = xc * cos(phi) - yc * sin(phi);
  out(1) = yc * cos(phi) + xc * sin(phi);
  return out;
}

Eigen::VectorXd UT_INES::Model2DAbsRel::ffull_OU(const Eigen::VectorXd& x) const {
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

Eigen::VectorXd UT_INES::Model2DAbsRel::GetInput(double t) const {
  Eigen::VectorXd out = RandomVector(getSw());
  out(0) += sin(t * 0.2);
  out(1) += cos(t * 0.5);
  return out;
}

Eigen::VectorXd UT_INES::Model2DAbsRel::InitialValue() const {
  Eigen::VectorXd x_true(11);
  x_true << 0.1, 0, 0., 0., 0., 0.5, 0.5, 0., 0., 0., 0.;
  return x_true;
}

ValWithCov UT_INES::Model2DAbsRel::InitialEstimation() const {
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
  return ValWithCov(InitialValue() + RandomVector(Sx), Sx);
}

Eigen::MatrixXd UT_INES::Model2DAbsRel::getSw() const {
  Eigen::MatrixXd Sw_ = Eigen::MatrixXd::Identity(8, 8) * 1;
  Sw_(0, 0) = 1.5;
  Sw_(1, 1) = 1.5;
  return Sw_;
}

Eigen::MatrixXd UT_INES::Model2DAbsRel::getSv() const {
  return Eigen::MatrixXd::Identity(6, 6) * 0.5;
}
