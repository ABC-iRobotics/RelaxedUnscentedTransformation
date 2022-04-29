#pragma once
#include "UTComponents.h"
#include "index_selector.h"

using namespace UTComponents;
using namespace RelaxedUnscentedTransformation;

typedef Eigen::LLT<Eigen::MatrixXd>::RealScalar Real;

Eigen::MatrixXd UTComponents::PartialChol(const Eigen::MatrixXd& a, const Eigen::VectorXi& inl) {
  const Eigen::Index n = a.rows();
  double eps = a.trace() * 1e-10;
  const Eigen::Index nOut = inl.size();
  Eigen::MatrixXd out = Eigen::MatrixXd(n, nOut);
  unsigned int nSkipped = 0;
  Eigen::VectorXi handled = Eigen::VectorXi::Zero(n);
  for (unsigned int i = 0; i < inl.size(); i++) {
	int j = inl(i);
	Real temp = a(j, j);
	for (unsigned int k = 0; k < i; k++)
	  temp -= out(j, k) * out(j, k);
	if (temp < eps && temp>-eps) {
	  nSkipped++;
	  break;
	}
	if (temp < Real(0)) {

	  std::stringstream errormsg;
	  errormsg << "The value " << temp << " is negative, when matrix (" << a <<
		") was  decomposed by column (" << inl.transpose() <<
		"). The matrix is not positive definite or numerical error.";
	  throw std::runtime_error(errormsg.str());
	}
	out(j, i) = sqrtl(temp);
	handled[j] = 1;

	for (unsigned int k = 0; k < n; k++)
	  if (k != j) {
		if (handled[k])
		  out(k, i) = 0;
		else {
		  temp = a(k, j);
		  for (unsigned int l = 0; l < i; l++)
			temp -= out(k, l) * out(j, l);
		  temp /= out(j, i);
		  out(k, i) = temp;
		}
	  }
  }
  return out.block(0, 0, n, nOut - nSkipped);
}

Eigen::MatrixXd UTComponents::FullChol(const Eigen::MatrixXd& a) {
  const Eigen::Index n = a.rows();
  double eps = a.trace() * 1e-10;
  const Eigen::Index nOut = n;
  Eigen::MatrixXd out = Eigen::MatrixXd(n, nOut);
  unsigned int nSkipped = 0;
  for (unsigned int j = 0; j < n; j++) {
	Real temp = a(j, j);
	for (unsigned int k = 0; k < j; k++)
	  temp -= out(j, k) * out(j, k);
	if (temp < eps && temp>-eps) {
	  nSkipped++;
	  break;
	}
	if (temp < Real(0)) {

	  std::stringstream errormsg;
	  errormsg << "The value " << temp << " is negative, when matrix (" << a <<
		") was  decomposed by all columns. The matrix is not positive definite or numerical error.";
	  throw std::runtime_error(errormsg.str());
	}
	for (unsigned int k = 0; k < j; k++)
	  out(k, j) = 0;
	out(j, j) = sqrtl(temp);
	for (unsigned int k = j + 1; k < n; k++) {
	  temp = a(k, j);
	  for (unsigned int l = 0; l < j; l++)
		temp -= out(k, l) * out(j, l);
	  temp /= out(j, j);
	  out(k, j) = temp;
	}
  }
  return out.block(0, 0, n, nOut - nSkipped);
}

std::vector<Eigen::VectorXd> UTComponents::GenSigmaDifferencesFull(const Eigen::MatrixXd& S) {
  // partial Choleski
  Eigen::MatrixXd L = FullChol(S);
  std::vector<Eigen::VectorXd> out;
  for (unsigned int l = 0; l < L.cols(); l++)
	out.push_back(L.col(l));
  return out;
}

#include <iostream>

std::vector<Eigen::VectorXd> UTComponents::GenSigmaDifferences(const Eigen::MatrixXd& S,
  const Eigen::VectorXi& inl) {
  // partial Choleski
  Eigen::MatrixXd L = PartialChol(S, inl);
  std::vector<Eigen::VectorXd> out;
  for (unsigned int l = 0; l < L.cols(); l++)
	out.push_back(L.col(l));
  return out;
}

ValWithCov UTComponents::LinearMappingOnb(const ValWithCov& b0_in, const Eigen::MatrixXd& F) {
  auto& b0 = b0_in.y;
  auto& Sb0 = b0_in.Sy;
  auto& Sxb0 = b0_in.Sxy;
  // Determine b related quantities
  Eigen::VectorXd b(b0.size() + F.rows());
  b.segment(0, b0.size()) = b0;
  b.segment(b0.size(), F.rows()) = F * b0;
  Eigen::MatrixXd Sb(b.size(), b.size());
  Sb.block(0, 0, b0.size(), b0.size()) = Sb0;
  {
	auto temp = F * Sb0;
	Sb.block(b0.size(), 0, F.rows(), b0.size()) = temp;
	Sb.block(0, b0.size(), b0.size(), F.rows()) = temp.transpose();
	Sb.block(b0.size(), b0.size(), F.rows(), F.rows()) = temp * F.transpose();
  }
  Eigen::MatrixXd Sxb(Sxb0.rows(), b.size());
  Sxb.block(0, 0, Sxb0.rows(), b0.size()) = Sxb0;
  Sxb.block(0, b0.size(), Sxb0.rows(), F.rows()) = Sxb0 * F.transpose();

  return ValWithCov(b, Sb, Sxb);
}

ValWithCov UTComponents::LinearMappingOnbWith0(const ValWithCov& b0_in, const Eigen::MatrixXd& F) {
  auto& b0 = b0_in.y;
  auto& Sb0 = b0_in.Sy;
  auto& Sxb0 = b0_in.Sxy;
  // Determine b related quantities
  Eigen::VectorXd b(1 + b0.size() + F.rows());
  b(0) = 0;
  b.segment(1, b0.size()) = b0;
  b.segment(1 + b0.size(), F.rows()) = F * b0;
  Eigen::MatrixXd Sb(b.size(), b.size());
  Sb.block(0, 0, 1, b.size()) = Eigen::VectorXd::Zero(b.size()).transpose();
  Sb.block(0, 0, b.size(), 1) = Eigen::VectorXd::Zero(b.size());
  Sb.block(1, 1, b0.size(), b0.size()) = Sb0;
  {
	auto temp = F * Sb0;
	Sb.block(1 + b0.size(), 1, F.rows(), b0.size()) = temp;
	Sb.block(1, 1 + b0.size(), b0.size(), F.rows()) = temp.transpose();
	Sb.block(1 + b0.size(), 1 + b0.size(), F.rows(), F.rows()) = temp * F.transpose();
  }
  Eigen::MatrixXd Sxb(Sxb0.rows(), b.size());
  Sxb.block(0, 0, Sxb0.rows(), 1) = Eigen::VectorXd::Zero(Sxb0.rows());
  Sxb.block(0, 1, Sxb0.rows(), b0.size()) = Sxb0;
  Sxb.block(0, 1 + b0.size(), Sxb0.rows(), F.rows()) = Sxb0 * F.transpose();

  return ValWithCov(b, Sb, Sxb);
}

ValWithCov UTComponents::MixedLinSourcesWithReordering(const ValWithCov& x0_in, const ValWithCov& b_in,
  const Eigen::VectorXi& il, const Eigen::MatrixXd& A, const Eigen::VectorXi& g) {
  auto& x0 = x0_in.y;
  auto& S0 = x0_in.Sy;
  auto& b = b_in.y;
  auto& Sb = b_in.Sy;
  auto& Sxb = b_in.Sxy;
  // Init x_l
  Eigen::VectorXd xl = VectorSelect(x0, il);
  // Determine y related quantities
  Eigen::VectorXd y = VectorSelect(b, g) + A * xl;

  Eigen::MatrixXd Sxbg = MatrixColumnSelect(Sxb, g);
  Eigen::MatrixXd Sbgbg = MatrixColumnSelect(MatrixRowSelect(Sb, g), g);
  Eigen::MatrixXd At = A.transpose();
  Eigen::MatrixXd Sxy = Sxbg + MatrixColumnSelect(S0, il) * At;
  Eigen::MatrixXd Sy = Sbgbg + A * MatrixRowSelect(Sxy, il) + MatrixColumnSelect(Sxbg.transpose(), il) * At;

  return ValWithCov(y, Sy, Sxy);
}
