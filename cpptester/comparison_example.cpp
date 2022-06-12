#include "UTCore.h"
#include "UTComponents.h"
#include <chrono>
#include <thread>
#include <iostream>

using namespace Eigen;
using namespace UT;

struct F1 {
  static Eigen::VectorXd f(const Eigen::VectorXd& x) {
	Eigen::VectorXd out(7);
	out(0) = sin(x(0) + 4. * x(1) - 0.5 * x(2));
	out(1) = cos(x(0) + 4. * x(1) - 0.5 * x(2));
	out(2) = x(3) + x(4);
	out(3) = x(3) + x(5);
	out(4) = x(3);
	out(5) = x(4);
	out(6) = x(5);
	return out;
  }

  ValWithCov UT(const ValWithCov& x) {
	auto xdiffs = GenSigmaDifferences(x.Sy);
	return UTCore(x.y, xdiffs, f, UTOriginal(xdiffs.size()));
  }
};

struct F2a {
  Eigen::MatrixXd A;
  Eigen::VectorXi il,inl;
  F2a() {
	A = Eigen::MatrixXd::Zero(7, 3);
	A(2, 0) = 1; A(2, 1) = 1;
	A(3, 0) = 1; A(3, 2) = 1;
	A(4, 0) = 1;
	A(5, 1) = 1;
	A(6, 2) = 1;
	il = Eigen::VectorXi(3);
	il(0) = 3;
	il(1) = 4;
	il(2) = 5;
	inl = Eigen::VectorXi(3);
	inl(0) = 0;
	inl(1) = 1;
	inl(2) = 2;
  }
  static Eigen::VectorXd f(const Eigen::VectorXd& x) {
	Eigen::VectorXd out(7);
	out(0) = sin(x(0) + 4. * x(1) - 0.5 * x(2));
	out(1) = cos(x(0) + 4. * x(1) - 0.5 * x(2));
	out(2) = 0;
	out(3) = 0;
	out(4) = 0;
	out(5) = 0;
	out(6) = 0;
	return out;
  }
  ValWithCov UT(const ValWithCov& x) {
	auto xdiffs = GenSigmaDifferences(x.Sy, inl);
	auto b = UTCore(x.y, xdiffs, f, UTOriginal(xdiffs.size()));
	return MixedLinSources(x, b, il, A);
  }
};

struct F2b {
  Eigen::MatrixXd A;
  Eigen::VectorXi il;
  ExactSubspace sp;
  double kappa;
  F2b(double kappa) : kappa(kappa) {
	A = Eigen::MatrixXd::Zero(7, 3);
	A(2, 0) = 1; A(2, 1) = 1;
	A(3, 0) = 1; A(3, 2) = 1;
	A(4, 0) = 1;
	A(5, 1) = 1;
	A(6, 2) = 1;
	il = Eigen::VectorXi(3);
	il(0) = 3;
	il(1) = 4;
	il(2) = 5;
	Eigen::VectorXi inl = Eigen::VectorXi(0);
	Eigen::VectorXi i(3);
	i(0) = 0;
	i(1) = 1;
	i(2) = 2;
	Eigen::VectorXd m(3);
	m(0) = 1.;
	m(1) = 4.;
	m(2) = -0.5;
	sp = ExactSubspace(6, inl, { {i,m} });
  }
  static Eigen::VectorXd f(const Eigen::VectorXd& x) {
	Eigen::VectorXd out(7);
	out(0) = sin(x(0) + 4. * x(1) - 0.5 * x(2));
	out(1) = cos(x(0) + 4. * x(1) - 0.5 * x(2));
	out(2) = 0;
	out(3) = 0;
	out(4) = 0;
	out(5) = 0;
	out(6) = 0;
	return out;
  }
  ValWithCov UT(const ValWithCov& x) {
	auto xdiffs = GenSigmaDifferences(x.Sy, sp);
	int m = xdiffs.size();
	auto b = UTCore(x.y, xdiffs, f, UTSettings(kappa, 1. - double(m) / kappa, 0.5 / kappa));
	return MixedLinSources(x, b, il, A);
  }
};
struct F3a {
  Eigen::MatrixXd A;
  Eigen::VectorXi il, inl, g;
  Eigen::MatrixXd F;
  F3a() {
	A = Eigen::MatrixXd::Zero(7, 3);
	A(2, 0) = 1; A(2, 1) = 1;
	A(3, 0) = 1; A(3, 2) = 1;
	A(4, 0) = 1;
	A(5, 1) = 1;
	A(6, 2) = 1;
	il = Eigen::VectorXi(3);
	il(0) = 3;
	il(1) = 4;
	il(2) = 5;
	inl = Eigen::VectorXi(3);
	inl(0) = 0;
	inl(1) = 1;
	inl(2) = 2;
	g = Eigen::VectorXi(7);
	g(0) = 1;
	g(1) = 2;
	g(2) = 0;
	g(3) = 0;
	g(4) = 0;
	g(5) = 0;
	g(6) = 0;
	F = Eigen::MatrixXd::Zero(0, 2);
  }
  static Eigen::VectorXd f(const Eigen::VectorXd& x) {
	Eigen::VectorXd out(2);
	out(0) = sin(x(0) + 4. * x(1) - 0.5 * x(2));
	out(1) = cos(x(0) + 4. * x(1) - 0.5 * x(2));
	return out;
  }
  ValWithCov UT(const ValWithCov& x) {
	auto xdiffs = GenSigmaDifferences(x.Sy, inl);
	auto b0 = UTCore(x.y, xdiffs, f, UTOriginal(xdiffs.size()));
	auto b = LinearMappingOnbWith0(b0, F);
	return MixedLinSources(x, Reordering(b, g), il, A);
  }
};
struct F3b {
  Eigen::MatrixXd A;
  Eigen::VectorXi il, g;
  ExactSubspace sp;
  Eigen::MatrixXd F;
  double kappa;
  F3b(double kappa) : kappa(kappa) {
	A = Eigen::MatrixXd::Zero(7, 3);
	A(2, 0) = 1; A(2, 1) = 1;
	A(3, 0) = 1; A(3, 2) = 1;
	A(4, 0) = 1;
	A(5, 1) = 1;
	A(6, 2) = 1;
	il = Eigen::VectorXi(3);
	il(0) = 3;
	il(1) = 4;
	il(2) = 5;
	Eigen::VectorXi inl = Eigen::VectorXi(0);
	Eigen::VectorXi i(3);
	i(0) = 0;
	i(1) = 1;
	i(2) = 2;
	Eigen::VectorXd m(3);
	m(0) = 1.;
	m(1) = 4.;
	m(2) = -0.5;
	sp = ExactSubspace(6, inl, { {i,m} });
	g = Eigen::VectorXi(7);
	g(0) = 1;
	g(1) = 2;
	g(2) = 0;
	g(3) = 0;
	g(4) = 0;
	g(5) = 0;
	g(6) = 0;
	F = Eigen::MatrixXd::Zero(0, 2);
  }
  static Eigen::VectorXd f(const Eigen::VectorXd& x) {
	Eigen::VectorXd out(2);
	out(0) = sin(x(0) + 4. * x(1) - 0.5 * x(2));
	out(1) = cos(x(0) + 4. * x(1) - 0.5 * x(2));
	return out;
  }
  ValWithCov UT(const ValWithCov& x) {
	auto xdiffs = GenSigmaDifferences(x.Sy, sp);
	int m = xdiffs.size();
	auto b0 = UTCore(x.y, xdiffs, f, UTSettings(kappa, 1. - double(m) / kappa, 0.5 / kappa));
	auto b = LinearMappingOnbWith0(b0, F);
	return MixedLinSources(x, Reordering(b,g), il, A);
  }
};
struct F4a {
  Eigen::MatrixXd A;
  Eigen::VectorXi il, inl;
  Eigen::MatrixXd F;
  F4a() {
	A = Eigen::MatrixXd::Zero(7, 3);
	A(2, 0) = 1; A(2, 1) = 1;
	A(3, 0) = 1; A(3, 2) = 1;
	A(4, 0) = 1;
	A(5, 1) = 1;
	A(6, 2) = 1;
	il = Eigen::VectorXi(3);
	il(0) = 3;
	il(1) = 4;
	il(2) = 5;
	inl = Eigen::VectorXi(3);
	inl(0) = 0;
	inl(1) = 1;
	inl(2) = 2;
	F = Eigen::MatrixXd::Zero(5, 2);
  }
  static Eigen::VectorXd f(const Eigen::VectorXd& x) {
	Eigen::VectorXd out(2);
	out(0) = sin(x(0) + 4. * x(1) - 0.5 * x(2));
	out(1) = cos(x(0) + 4. * x(1) - 0.5 * x(2));
	return out;
  }
  ValWithCov UT(const ValWithCov& x) {
	auto xdiffs = GenSigmaDifferences(x.Sy, inl);
	auto b0 = UTCore(x.y, xdiffs, f, UTOriginal(xdiffs.size()));
	auto b = LinearMappingOnb(b0, F);
	return MixedLinSources(x, b, il, A);
  }
};
struct F4b {
  Eigen::MatrixXd A;
  Eigen::VectorXi il;
  ExactSubspace sp;
  Eigen::MatrixXd F;
  F4b() {
	A = Eigen::MatrixXd::Zero(7, 3);
	A(2, 0) = 1; A(2, 1) = 1;
	A(3, 0) = 1; A(3, 2) = 1;
	A(4, 0) = 1;
	A(5, 1) = 1;
	A(6, 2) = 1;
	il = Eigen::VectorXi(3);
	il(0) = 3;
	il(1) = 4;
	il(2) = 5;
	Eigen::VectorXi inl = Eigen::VectorXi(0);
	Eigen::VectorXi i(3);
	i(0) = 0;
	i(1) = 1;
	i(2) = 2;
	Eigen::VectorXd m(3);
	m(0) = 1.;
	m(1) = 4.;
	m(2) = -0.5;
	sp = ExactSubspace(6, inl, { {i,m} });
	F = Eigen::MatrixXd::Zero(5, 2);
  }
  static Eigen::VectorXd f(const Eigen::VectorXd& x) {
	Eigen::VectorXd out(2);
	out(0) = sin(x(0) + 4. * x(1) - 0.5 * x(2));
	out(1) = cos(x(0) + 4. * x(1) - 0.5 * x(2));
	return out;
  }
  ValWithCov UT(const ValWithCov& x) {
	auto xdiffs = GenSigmaDifferences(x.Sy, sp);
	auto b0 = UTCore(x.y, xdiffs, f, UTOriginal(xdiffs.size()));
	auto b = LinearMappingOnb(b0, F);
	return MixedLinSources(x, b, il, A);
  }
};

ValWithCov GenInput(double traceS, ValWithCov& y) {
  Eigen::VectorXd x = Eigen::VectorXd::Random(6);
  Eigen::MatrixXd sqrtS = Eigen::MatrixXd::Random(6, 6);
  Eigen::MatrixXd S = sqrtS * sqrtS.transpose();
  sqrtS *= sqrt(abs(traceS / S.trace()));
  S *= traceS / S.trace();
  {
	std::vector<Eigen::VectorXd> out;
	std::vector<Eigen::VectorXd> in;
	int N = 1e5;
	y.y = Eigen::VectorXd::Zero(7);
	for (int i = 0; i < N; i++) {
	  Eigen::VectorXd x_ = RandomVector(sqrtS) + x;
	  in.push_back(x_);
	  auto y_ = F1::f(x_);
	  out.push_back(y_);
	  y.y += y_;
	}
	y.y /= double(N);
	y.Sy = Eigen::MatrixXd::Zero(7, 7);
	y.Sxy = Eigen::MatrixXd::Zero(6, 7);
	for (int i = 0; i < N; i++) {
	  Eigen::VectorXd d = out[i] - y.y;
	  y.Sy += d * d.transpose();
	  y.Sxy += (in[i] - x) * d.transpose();
	}
	y.Sy /= double(N - 1);
	y.Sxy /= double(N - 1);
  }
  return ValWithCov(x, S);
}

void printV(std::string str, const ValWithCov& v) {
  std::cout << str << v.y.transpose() << std::endl << std::endl << v.Sy << std::endl << std::endl << v.Sxy << std::endl << std::endl;
}

template<typename Func>
double measureTime(int N, Func f) {
  ValWithCov y, x(Eigen::VectorXd::Zero(6), Eigen::MatrixXd::Identity(6, 6));
  std::this_thread::sleep_for(std::chrono::seconds(5));

  auto start = std::chrono::system_clock::now();
  for (int n = 0; n < N; n++)
	y = f.UT(x);
  auto end = std::chrono::system_clock::now();
  auto dur_original = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  return (double)dur_original / double(N);
}

template<typename F>
void gendiff(const ValWithCov& x, const ValWithCov& ytrue, F f, std::vector<double>& dy, std::vector<double>& dSy) {
  auto y = f.UT(x);
  //printV("true: ", ytrue);
  //printV("est: ", y);
  dy.push_back((y.y - ytrue.y).squaredNorm());
  dSy.push_back((y.Sy - ytrue.Sy).squaredNorm());
}

double mean(const std::vector<double>& v) {
  double out = 0;
  for (auto it : v)
	out += it;
  return out / double(v.size());
}

int main() {
  // Determine computational time
  if (0) {
	int N = 1e7;
	F1 f1; F2a f2a; F2b f2b(1); F3a f3a; F3b f3b(1); F4a f4a; F4b f4b;

	std::cout << "f1: " << measureTime(N, f1) << std::endl;
	std::cout << "f2a: " << measureTime(N, f2a) << std::endl;
	std::cout << "f2b: " << measureTime(N, f2b) << std::endl;
	std::cout << "f3a: " << measureTime(N, f3a) << std::endl;
	std::cout << "f3b: " << measureTime(N, f3b) << std::endl;
	std::cout << "f4a: " << measureTime(N, f4a) << std::endl;
	std::cout << "f4b: " << measureTime(N, f4b) << std::endl;
	std::cout << "f1: " << measureTime(N, f1) << std::endl;
	std::cout << "f2a: " << measureTime(N, f2a) << std::endl;
	std::cout << "f2b: " << measureTime(N, f2b) << std::endl;
  }

  // Check accuracy
  if (1) {
	int N = 1e4;
	auto traceSvec = {0.1, 1., 10.};
	F1 f1; F2a f2a; F2b f2b1(1), f2b2(2), f2b3(3); F3a f3a; F3b f3b1(1), f3b2(2), f3b3(3);

	for (auto traceS : traceSvec) {
	  std::vector<double> dy1, dy2a, dy2b1, dy2b2, dy2b3, dy3a, dy3b1, dy3b2, dy3b3;
	  std::vector<double> dSy1, dSy2a, dSy2b1, dSy2b2, dSy2b3, dSy3a, dSy3b1, dSy3b2, dSy3b3;

	  for (int n = 0; n < N; n++) {
		//std::cout << n << std::endl;
		ValWithCov ytrue;
		auto x = GenInput(traceS, ytrue);
		gendiff(x, ytrue, f1, dy1, dSy1);
		gendiff(x, ytrue, f2a, dy2a, dSy2a);
		gendiff(x, ytrue, f2b1, dy2b1, dSy2b1);
		gendiff(x, ytrue, f2b2, dy2b2, dSy2b2);
		gendiff(x, ytrue, f2b3, dy2b3, dSy2b3);
		gendiff(x, ytrue, f3a, dy3a, dSy3a);
		gendiff(x, ytrue, f3b1, dy3b1, dSy3b1);
		gendiff(x, ytrue, f3b2, dy3b2, dSy3b2);
		gendiff(x, ytrue, f3b3, dy3b3, dSy3b3);
	  }
	  std::cout << "traceS: " << traceS << std::endl;
	  std::cout << "1: " << mean(dy1) << " " << mean(dSy1) << std::endl;
	  std::cout << "2a: " << mean(dy2a) << " " << mean(dSy2a) << std::endl;
	  std::cout << "2b1: " << mean(dy2b1) << " " << mean(dSy2b1) << std::endl;
	  std::cout << "2b2: " << mean(dy2b2) << " " << mean(dSy2b2) << std::endl;
	  std::cout << "2b3: " << mean(dy2b3) << " " << mean(dSy2b3) << std::endl;
	  std::cout << "3a: " << mean(dy3a) << " " << mean(dSy3a) << std::endl;
	  std::cout << "3b1: " << mean(dy3b1) << " " << mean(dSy3b1) << std::endl;
	  std::cout << "3b2: " << mean(dy3b2) << " " << mean(dSy3b2) << std::endl;
	  std::cout << "3b3: " << mean(dy3b3) << " " << mean(dSy3b3) << std::endl << std::endl;
	}
  }
  return 0;
	
}