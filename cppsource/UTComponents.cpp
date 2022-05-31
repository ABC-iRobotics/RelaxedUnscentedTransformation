#include "UTComponents.h"
#include "Cholesky.h"
#include "IndexSelector.h"
using namespace UT;

typedef Eigen::LLT<Eigen::MatrixXd>::RealScalar Real;

std::vector<Eigen::VectorXd> UT::GenSigmaDifferencesFull(const Eigen::MatrixXd& S) {
  // partial Choleski
  Eigen::MatrixXd L = FullChol(S);
  std::vector<Eigen::VectorXd> out;
  for (unsigned int l = 0; l < L.cols(); l++)
	out.push_back(L.col(l));
  return out;
}

std::vector<Eigen::VectorXd> UT::GenSigmaDifferences(const Eigen::MatrixXd& S,
  const Eigen::VectorXi& inl) {
  // partial Choleski
  Eigen::MatrixXd L = PartialChol(S, inl);
  std::vector<Eigen::VectorXd> out;
  for (unsigned int l = 0; l < L.cols(); l++)
	out.push_back(L.col(l));
  return out;
}

std::vector<Eigen::VectorXd> UT::GenSigmaDifferencesFromExactSubspace(const Eigen::MatrixXd& S, const ExactSubspace& sp) {
  Eigen::MatrixXd Stemp = sp.Q * S * sp.Q1().transpose();
  // partial Choleski
  Eigen::MatrixXd L = UT::PartialChol(Stemp, Eigen::VectorXi::LinSpaced(sp.m, 0, sp.m - 1)); //TODO: test!!
  L = sp.Q.transpose() * L;
  std::vector<Eigen::VectorXd> out;
  for (unsigned int l = 0; l < L.cols(); l++)
	out.push_back(L.col(l));
  return out;
}

ValWithCov UT::LinearMappingOnb(const ValWithCov& b0, const Eigen::MatrixXd& F) {
  auto& b0_ = b0.y;
  auto& Sb0 = b0.Sy;
  auto& Sxb0 = b0.Sxy;
  // Determine b related quantities
  Eigen::VectorXd b(b0_.size() + F.rows());
  b.segment(0, b0_.size()) = b0_;
  b.segment(b0_.size(), F.rows()) = F * b0_;
  Eigen::MatrixXd Sb(b.size(), b.size());
  Sb.block(0, 0, b0_.size(), b0_.size()) = Sb0;
  {
	auto temp = F * Sb0;
	Sb.block(b0_.size(), 0, F.rows(), b0_.size()) = temp;
	Sb.block(0, b0_.size(), b0_.size(), F.rows()) = temp.transpose();
	Sb.block(b0_.size(), b0_.size(), F.rows(), F.rows()) = temp * F.transpose();
  }
  Eigen::MatrixXd Sxb(Sxb0.rows(), b.size());
  Sxb.block(0, 0, Sxb0.rows(), b0_.size()) = Sxb0;
  Sxb.block(0, b0_.size(), Sxb0.rows(), F.rows()) = Sxb0 * F.transpose();

  return ValWithCov(b, Sb, Sxb);
}

ValWithCov UT::LinearMappingOnbWith0(const ValWithCov& b0, const Eigen::MatrixXd& F) {
  auto& b0_ = b0.y;
  auto& Sb0 = b0.Sy;
  auto& Sxb0 = b0.Sxy;
  // Determine b related quantities
  Eigen::VectorXd b(1 + b0_.size() + F.rows());
  b(0) = 0;
  b.segment(1, b0_.size()) = b0_;
  b.segment(1 + b0_.size(), F.rows()) = F * b0_;
  Eigen::MatrixXd Sb(b.size(), b.size());
  Sb.block(0, 0, 1, b.size()) = Eigen::VectorXd::Zero(b.size()).transpose();
  Sb.block(0, 0, b.size(), 1) = Eigen::VectorXd::Zero(b.size());
  Sb.block(1, 1, b0_.size(), b0_.size()) = Sb0;
  {
	auto temp = F * Sb0;
	Sb.block(1 + b0_.size(), 1, F.rows(), b0_.size()) = temp;
	Sb.block(1, 1 + b0_.size(), b0_.size(), F.rows()) = temp.transpose();
	Sb.block(1 + b0_.size(), 1 + b0_.size(), F.rows(), F.rows()) = temp * F.transpose();
  }
  Eigen::MatrixXd Sxb(Sxb0.rows(), b.size());
  Sxb.block(0, 0, Sxb0.rows(), 1) = Eigen::VectorXd::Zero(Sxb0.rows());
  Sxb.block(0, 1, Sxb0.rows(), b0_.size()) = Sxb0;
  Sxb.block(0, 1 + b0_.size(), Sxb0.rows(), F.rows()) = Sxb0 * F.transpose();

  return ValWithCov(b, Sb, Sxb);
}

ValWithCov UT::MixedLinSourcesWithReordering(const ValWithCov& x, const ValWithCov& b,
  const Eigen::VectorXi& il, const Eigen::MatrixXd& A, const Eigen::VectorXi& g) {
  auto& x0_ = x.y;
  auto& S0 = x.Sy;
  auto& b_ = b.y;
  auto& Sb = b.Sy;
  auto& Sxb = b.Sxy;
  // Init x_l
  Eigen::VectorXd xl = VectorSelect(x0_, il);
  // Determine y related quantities
  Eigen::VectorXd y = VectorSelect(b_, g) + A * xl;

  Eigen::MatrixXd Sxbg = MatrixColumnSelect(Sxb, g);
  Eigen::MatrixXd Sbgbg = MatrixColumnSelect(MatrixRowSelect(Sb, g), g);
  Eigen::MatrixXd At = A.transpose();
  Eigen::MatrixXd Sxy = Sxbg + MatrixColumnSelect(S0, il) * At;
  Eigen::MatrixXd Sy = Sbgbg + A * MatrixRowSelect(Sxy, il) + MatrixColumnSelect(Sxbg.transpose(), il) * At;

  return ValWithCov(y, Sy, Sxy);
}

ValWithCov UT::MixedLinSources(const ValWithCov& x, const ValWithCov& b,
  const Eigen::VectorXi& il, const Eigen::MatrixXd& A) {
  auto& x0_ = x.y;
  auto& S0 = x.Sy;
  auto& b_ = b.y;
  auto& Sb = b.Sy;
  auto& Sxb = b.Sxy;
  // Init x_l
  Eigen::VectorXd xl = VectorSelect(x0_, il);
  // Determine y related quantities
  Eigen::VectorXd y = b_ + A * xl;

  Eigen::MatrixXd At = A.transpose();
  Eigen::MatrixXd Sxy = Sxb + MatrixColumnSelect(S0, il) * At;
  Eigen::MatrixXd Sy = Sb + A * MatrixRowSelect(Sxy, il) + MatrixColumnSelect(Sxb.transpose(), il) * At;

  return ValWithCov(y, Sy, Sxy);
}

UT::ExactSubspace::ExactSubspace(int n, const Eigen::VectorXi& inl, const MixedNonlinearityList& mix) {
  // Construct M matrix from the weights keeping it orthogonal to inl
  typedef Eigen::MatrixXd Matrix;
  Matrix M = Matrix::Zero(mix.size() + inl.size(), n);
  for (int n = 0; n < inl.size(); n++)
	M(n, inl(n)) = 1;
  for (int n = 0; n < mix.size(); n++)
	for (int i0 = 0; i0 < mix[n].i.size(); i0++) {
	  bool flag = false;
	  for (int i = 0; i < inl.size(); i++)
		if (mix[n].i[i0] == inl(i))
		  flag = true;
	  if (!flag)
		M(n + inl.size(), mix[n].i[i0]) = mix[n].M(i0);
	}
  // RQ factorization
  Eigen::HouseholderQR<Matrix> solver(M.transpose());
  Eigen::MatrixXd Qdense = solver.householderQ().transpose();
  double rel_treshold = 1e-6;
  // Compute rank
  m = n;
  {
	Eigen::MatrixXd to_compute_rank = (M * Qdense.transpose()).cwiseAbs();
	double treshold = to_compute_rank.maxCoeff() * rel_treshold;
	for (int i = 0; i < to_compute_rank.cols(); i++)
	  if (to_compute_rank.col(i).maxCoeff() < treshold) {
		m = i;
		break;
	  }
  }
  // make it sparse
  for (int i = 0; i < Qdense.rows(); i++)
	for (int j = 0; j < Qdense.cols(); j++)
	  if (abs(Qdense(i, j)) < 1e-8)
		Qdense(i, j) = 0;
  // save it sparse
  Q = Qdense.sparseView();
}
