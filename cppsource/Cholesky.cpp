#include "Cholesky.h"
#include <iostream>

typedef Eigen::LLT<Eigen::MatrixXd>::RealScalar Real;

Eigen::MatrixXd UT::PartialChol(const Eigen::MatrixXd& a, const Eigen::VectorXi& inl) {
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
	  std::cout << "The value " << temp << " is negative, when matrix (" << a <<
		") was  decomposed by column (" << inl.transpose() <<
		"). The matrix is not positive definite or numerical error." << std::endl;
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

Eigen::MatrixXd UT::FullChol(const Eigen::MatrixXd& a) {
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
