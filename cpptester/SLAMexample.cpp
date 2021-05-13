#include "SLAMexample.h"
#include "defs.h"
#include <iostream>
#include "SLAMmodels.h"
#include "UT.h"

using namespace Eigen;

void SLAM_state_example() {
	double Ts = 0.002;

	int N = 5;
	auto stateUpdate = SLAMStateUpdate();

	Eigen::VectorXd x0 = Eigen::VectorXd::Ones(2 * N + 3);

	double v = 0.4, Sv = 1;;
	double omega = 0.2, Somega = 1;
	MatrixXd Sx0 = MatrixXd::Identity(2 * N + 3, 2 * N + 3) / 2.;
	Sx0(2, 2) = 0.02;

	Eigen::VectorXd y1;
	Eigen::MatrixXd Sy1, Sxy1;
	long N_ = 100000;
	{
		auto start = std::chrono::system_clock::now();
		for (long i = 0; i < N_; i++)
			stateUpdate.UT(Ts, v, omega, Sv, Somega, x0, Sx0, y1, Sy1, Sxy1);
		auto end = std::chrono::system_clock::now();
		unsigned long dur = (unsigned long)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		printf("new UT : %ld \n", dur / 1000);
	}
	Eigen::VectorXd y2;
	Eigen::MatrixXd Sy2, Sxy2;
	{
		Eigen::VectorXd in(2 * N + 5);
		in(0) = v;
		in(1) = omega;
		in.segment(2, 2 * N + 3) = x0;
		Eigen::MatrixXd Sin = Eigen::MatrixXd::Zero(2 * N + 5, 2 * N + 5);
		Sin(0, 0) = Sv;
		Sin(1, 1) = Somega;
		Sin.block(2, 2, 2 * N + 3, 2 * N + 3) = Sx0;

		auto start = std::chrono::system_clock::now();
		for (long i = 0; i < N_; i++)
			UT(in, Sin, [Ts](const Eigen::VectorXd& in)->Eigen::VectorXd { return SLAMStateUpdateFull(in, Ts); }, y2, Sy2, Sxy2);
		auto end = std::chrono::system_clock::now();
		unsigned long dur = (unsigned long)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		printf("pure UT : %ld \n", dur / 1000);
	}
	
	std::cout << y1 << std::endl << std::endl;
	std::cout << y2 << std::endl << std::endl;
	std::cout << y1 - y2 << std::endl << std::endl;
	std::cout << Sy1 << std::endl << std::endl;
	std::cout << Sy2 << std::endl << std::endl;
	std::cout << Sy1 - Sy2 << std::endl << std::endl;
	std::cout << Sxy1 << std::endl << std::endl;
	std::cout << Sxy2 << std::endl << std::endl;
	std::cout << Sxy1 - Sxy2 << std::endl;
	
}

void SLAM_output_example() {
	int N = 7;
	std::vector<int> actives = { 0, 1, 5 };
	auto stateUpdate = SLAMOutputUpdate();
	
	auto slamFullModel = SLAM_output_full_fcn(actives);
	
	Eigen::VectorXd x0 = Eigen::VectorXd::Ones(2 * N + 3);
	x0(0) = -2;
	x0(3) = 5;

	//std::cout << slamFullModel(x0) << std::endl << std::endl;

	MatrixXd Sx0 = MatrixXd::Identity(2 * N + 3, 2 * N + 3) / 4.;
	Sx0(2, 2) = 0.02;
	Eigen::VectorXd y1;
	Eigen::MatrixXd Sy1, Sxy1;
	long N_ = 1000000;
	{
		auto start = std::chrono::system_clock::now();
		for (long i = 0; i < N_; i++)
			stateUpdate.UT(actives, x0, Sx0, y1, Sy1, Sxy1);
		auto end = std::chrono::system_clock::now();
		unsigned long dur = (unsigned long)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		printf("new UT : %ld \n", dur / 1000);
	}
	
	Eigen::VectorXd y2;
	Eigen::MatrixXd Sy2, Sxy2;
	{
		auto start = std::chrono::system_clock::now();
		for (long i = 0; i < N_; i++)
			UT(x0, Sx0, slamFullModel, y2, Sy2, Sxy2);
		auto end = std::chrono::system_clock::now();
		unsigned long dur = (unsigned long)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		printf("pure UT : %ld \n", dur / 1000);
	}

	std::cout << y1 << std::endl << std::endl;
	std::cout << y2 << std::endl << std::endl;
	std::cout << y1 - y2 << std::endl << std::endl;
	std::cout << Sy1 << std::endl << std::endl;
	std::cout << Sy2 << std::endl << std::endl;
	std::cout << Sy1 - Sy2 << std::endl << std::endl;
	std::cout << Sxy1 << std::endl << std::endl;
	std::cout << Sxy2 << std::endl << std::endl;
	std::cout << Sxy1 - Sxy2 << std::endl;
}

std::vector<std::pair<int, std::pair<double,double>>> SLAM_stateupdate_data() {
	std::vector<std::pair<int, std::pair<double, double>>> out;
	double Ts = 0.01;
	int K = 52;
	for (int N = 1; N < K; N=N+2) {
		//std::cout << N << std::endl;
		auto stateUpdate = SLAMStateUpdate();
		Eigen::VectorXd u(2);
		double v = 1;
		double omega = 2;
		u(0) = v;
		u(1) = omega;
		Eigen::VectorXd x0 = Eigen::VectorXd::Ones(2 * N + 3);
		Eigen::VectorXd in(2 + 2 * N + 3);
		in.segment(0, 2) = u;
		in.segment(2, 3 + 2 * N) = x0;
		MatrixXd Su = MatrixXd::Identity(2, 2);
		double Sv = 0.4;
		double Somega = 0.2;
		Su(0, 0) = Sv;
		Su(1, 1) = Somega;
		MatrixXd Sx0 = MatrixXd::Identity(2 * N + 3, 2 * N + 3) / 2.;
		Sx0(2, 2) = 0.02;
		MatrixXd Sin = MatrixXd::Zero(2 * N + 5, 2 * N + 5);
		Sin.block(0, 0, 2, 2) = Su;
		Sin.block(2, 2, 2 * N + 3, 2 * N + 3) = Sx0;

		Eigen::VectorXd y1;
		Eigen::MatrixXd Sy1, Sxy1;
		long N_ = 10000;
		double dur_new_ms, dur_old_ms;
		{
			auto start = std::chrono::system_clock::now();
			for (long i = 0; i < N_; i++)
				stateUpdate.UT(Ts, v, omega, Sv, Somega, x0, Sx0, y1, Sy1, Sxy1);
			auto end = std::chrono::system_clock::now();
			dur_new_ms = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;
		}
		Eigen::VectorXd y2;
		Eigen::MatrixXd Sy2, Sxy2;
		{
			auto start = std::chrono::system_clock::now();
			for (long i = 0; i < N_; i++)
				UT(in, Sin, [Ts](const VectorXd& in)->VectorXd { return SLAMStateUpdateFull(in, Ts); }, y2, Sy2, Sxy2);
			auto end = std::chrono::system_clock::now();
			dur_old_ms = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;
		}

		out.push_back({ N,{dur_old_ms,dur_new_ms} });
	}
	return out;
}

void to_compute() {
	auto  data = SLAM_stateupdate_data();

	std::cout << "N:" << std::endl;
	for (auto it : data)
		std::cout << it.first << " ";
	std::cout << std::endl << "old [ms]:" << std::endl;
	for (auto it : data)
		std::cout << it.second.first << " ";
	std::cout << std::endl << "new [ms]:" << std::endl;
	for (auto it : data)
		std::cout << it.second.second << " ";
	std::cout << std::endl;
}

void SLAM_outputupdate_data(Eigen::VectorXi& Nout, Eigen::VectorXi&Naout, Eigen::MatrixXd& out_old, Eigen::MatrixXd& out_new) {
	int K = 25;
	int Ka = 15;
	Nout = Eigen::VectorXi(K);
	Naout = Eigen::VectorXi(Ka);
	out_old = Eigen::MatrixXd(K, Ka);
	out_new = Eigen::MatrixXd(K, Ka);

	for (int n0 = 0; n0 < K; n0++) {
		int N = 2 * n0 + 1;
		Nout(n0) = N;
		for (int na0 = 0; na0 < Ka; na0++) {
			int Na = 2 * na0 + 1;
			Naout(na0) = Na;
			//std::cout << "N/Na: " << N << "/" << Na << std::endl;
			if (Na > N) {
				out_new(n0, na0) = -1;
				out_old(n0, na0) = -1;
				continue;
			}

			std::vector<int> actives;
			for (int n = 0; n < Na; n++)
				actives.push_back(n);
			auto stateUpdate = SLAMOutputUpdate();

			auto slamFullModel = SLAM_output_full_fcn(actives);

			Eigen::VectorXd x0 = Eigen::VectorXd::Ones(2 * N + 3);
			for (int n = 0; n < N; n++)
				x0(3 + n * 2) = 5;

			MatrixXd Sx0 = MatrixXd::Identity(2 * N + 3, 2 * N + 3) / 4.;
			Sx0(2, 2) = 0.02;

			Eigen::VectorXd y1;
			Eigen::MatrixXd Sy1, Sxy1;
			long N_ = 10000;
			{
				auto start = std::chrono::system_clock::now();
				for (long i = 0; i < N_; i++)
					stateUpdate.UT(actives, x0, Sx0, y1, Sy1, Sxy1);
				auto end = std::chrono::system_clock::now();
				out_new(n0, na0) = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000;
			}

			Eigen::VectorXd y2;
			Eigen::MatrixXd Sy2, Sxy2;
			{
				auto start = std::chrono::system_clock::now();
				for (long i = 0; i < N_; i++)
					UT(x0, Sx0, slamFullModel, y2, Sy2, Sxy2);
				auto end = std::chrono::system_clock::now();
				out_old(n0, na0) = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
			}
		}
	}
	//std::cout << "new:" << std::endl << out_new << std::endl << std::endl;
	//std::cout << "old:" << std::endl << out_old << std::endl << std::endl;
}

