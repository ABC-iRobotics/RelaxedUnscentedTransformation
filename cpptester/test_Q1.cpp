#include "common/unity.h"
void setUp() {}
void tearDown() {}

#include "defs.h"
#include "SLAMexample.h"
#include "lin_example.h"
#include <iostream>
#include <fstream>
#include "readData.h"

#include "Simulator.h"
#include "AUT.h"
#include "RelAUT.h"
using namespace Eigen;

void genQ1data() {
	/*
	{
		std::string filename = "RawCPUtimes_";
		std::ofstream logfile;
		logfile.open(filename + std::to_string(std::chrono::system_clock::now().time_since_epoch().count())
			+ ".m", std::ios::out | std::ios::trunc);
		// CPUtime of SLAM state update
		{
			auto data = SLAM_stateupdate_data();
			logfile << "SLAM_stateupdate_N_CPUtime_UT = [...\n";
			for (auto it : data)
				logfile << it.first << " ";
			logfile << ";...\n";
			for (auto it : data)
				logfile << it.second.first << " ";
			logfile << "];\nSLAM_stateupdate_N_CPUtime_newUT = [...\n";
			for (auto it : data)
				logfile << it.first << " ";
			logfile << ";...\n";
			for (auto it : data)
				logfile << it.second.second << " ";
			logfile << "];\n\n";
		}
		// CPUtime of SLAM output update
		{
			Eigen::VectorXi N;
			Eigen::VectorXi Na;
			Eigen::MatrixXd out_old;
			Eigen::MatrixXd out_new;
			SLAM_outputupdate_data(N, Na, out_old, out_new);
			logfile << "SLAM_outputupdate_CPUtime_N = [ " << N.transpose() << "];\n";
			logfile << "SLAM_outputupdate_CPUtime_Na = [ " << Na.transpose() << "];\n";
			logfile << "SLAM_outputupdate_CPUtime_UT = [...\n";
			for (int i = 0; i < out_old.rows(); i++)
				logfile << out_old.row(i) << (i + 1 == out_old.rows() ? "];\n" : ";...\n");
			logfile << "SLAM_outputupdate_CPUtime_newUT = [...\n";
			for (int i = 0; i < out_new.rows(); i++)
				logfile << out_new.row(i) << (i + 1 == out_new.rows() ? "];\n" : ";...\n");
		}
		logfile.close();
	}
	*/
	// Simulation results
	long time_start_sec = 1248272272, time_start_msec = 841;
	double simulation_duration_sec = 80;
	SimSettings set(time_start_sec, time_start_msec, simulation_duration_sec);
	ConvertGroundTruthData(time_start_sec, time_start_msec, simulation_duration_sec);
	// Simple prediction without adding landmarks
	{
		set.name = "UT_nolandmarks";
		set.cType = SimSettings::UT;
		set.enableAddingLandmarks = false;
		set.enableKalmanFiltering = false;
		set.useAdaptive = false;
		Simulator sim(set);
		sim.Run(Simulator::LOGPATH);
		sim.Run(Simulator::TIME_MEASURE);
	}
	{
		set.name = "newUT_nolandmarks";
		set.cType = SimSettings::NewUT;
		set.enableAddingLandmarks = false;
		set.enableKalmanFiltering = false;
		set.useAdaptive = false;
		Simulator sim(set);
		sim.Run(Simulator::LOGPATH);
		sim.Run(Simulator::TIME_MEASURE);
	}
	
	// Kalman-filtering
	{
		set.name = "UKF";
		set.cType = SimSettings::UT;
		set.enableAddingLandmarks = true;
		set.enableKalmanFiltering = true;
		set.useAdaptive = false;
		Simulator sim(set);
		sim.Run(Simulator::LOGPATH);
		sim.Run(Simulator::TIME_MEASURE);
	}
	{
		set.name = "newUKF";
		set.cType = SimSettings::NewUT;
		set.enableAddingLandmarks = true;
		set.enableKalmanFiltering = true;
		set.useAdaptive = false;
		Simulator sim(set);
		sim.Run(Simulator::LOGPATH);
		sim.Run(Simulator::TIME_MEASURE);
	}

	// Simple prediction with adding landmarks
	{
		set.name = "UT_withlandmarks";
		set.cType = SimSettings::UT;
		set.enableAddingLandmarks = true;
		set.enableKalmanFiltering = false;
		set.useAdaptive = false;
		Simulator sim(set);
		sim.Run(Simulator::LOGPATH);
		sim.Run(Simulator::TIME_MEASURE);
	}
	{
		set.name = "newUT_withlandmarks";
		set.cType = SimSettings::NewUT;
		set.enableAddingLandmarks = true;
		set.enableKalmanFiltering = false;
		set.useAdaptive = false;
		Simulator sim(set);
		sim.Run(Simulator::LOGPATH);
		sim.Run(Simulator::TIME_MEASURE);
	}
	{
		set.name = "AUKF";
		set.cType = SimSettings::UT;
		set.enableAddingLandmarks = true;
		set.enableKalmanFiltering = true;
		set.useAdaptive = true;
		Simulator sim(set);
		sim.Run(Simulator::LOGPATH);
		sim.Run(Simulator::TIME_MEASURE);
	}
	{
		set.name = "newAUKF";
		set.cType = SimSettings::NewUT;
		set.enableAddingLandmarks = true;
		set.enableKalmanFiltering = true;
		set.useAdaptive = true;
		Simulator sim(set);
		sim.Run(Simulator::LOGPATH);
		sim.Run(Simulator::TIME_MEASURE);
	}
}


int main (void) {

	genQ1data();


	return 0;

	/*
	MatrixXd S0 = MatrixXd::Zero(11, 11);
	S0 << 0.0320, 0.0196, 0.0214, 0.0142, 0.0127, 0.0276, 0.0155, 0.0144, 0.0198, 0.0269, 0.0237,
		0.0196, 0.0481, 0.0237, 0.0256, 0.0189, 0.0369, 0.0197, 0.0235, 0.0249, 0.0338, 0.0300,
		0.0214, 0.0237, 0.0358, 0.0239, 0.0162, 0.0362, 0.0207, 0.0222, 0.0253, 0.0340, 0.0300,
		0.0142, 0.0256, 0.0239, 0.0668, 0.0154, 0.0323, 0.0250, 0.0234, 0.0285, 0.0294, 0.0245,
		0.0127, 0.0189, 0.0162, 0.0154, 0.0217, 0.0258, 0.0168, 0.0174, 0.0171, 0.0159, 0.0189,
		0.0276, 0.0369, 0.0362, 0.0323, 0.0258, 0.0570, 0.0290, 0.0322, 0.0327, 0.0412, 0.0360,
		0.0155, 0.0197, 0.0207, 0.0250, 0.0168, 0.0290, 0.0266, 0.0196, 0.0246, 0.0229, 0.0215,
		0.0144, 0.0235, 0.0222, 0.0234, 0.0174, 0.0322, 0.0196, 0.1889, 0.0215, 0.0316, 0.0265,
		0.0198, 0.0249, 0.0253, 0.0285, 0.0171, 0.0327, 0.0246, 0.0215, 0.0409, 0.0310, 0.0253,
		0.0269, 0.0338, 0.0340, 0.0294, 0.0159, 0.0412, 0.0229, 0.0316, 0.0310, 0.0657, 0.0446,
		0.0237, 0.0300, 0.0300, 0.0245, 0.0189, 0.0360, 0.0215, 0.0265, 0.0253, 0.0446, 0.0574;

	//auto bl = Block2(S0.derived(), 0, 0, 3, 4);

	auto bl = S0.block(0, 0, 3, 4);
	auto bl2 = S0.block2(0, 0, 3, 4);
	Eigen::MatrixXd bl3 = bl;
	std::cout << bl << std::endl;
	std::cout << bl2 << std::endl;

	std::cout << bl * bl.transpose() - bl2 * bl.transpose() << std::endl;

	return 0;*/
	//lin_example();

	

	//SLAM_outputupdate_data();

	/*
	auto out = ReadOdometryData();
	std::cout << out.size() << std::endl;
	int n = out.size()-1;
	auto last = out[n];
	std::cout << last.t0 << "." << last.t1 << " " << last.v << " " << last.omega << std::endl;
	/*
	// Deriving matrix A
	MatrixXd A = MatrixXd::Zero(11, 6);
	for (int i = 0; i < 6; i++)
		A(i, i) = 1;
	A(6, 2) = Ts;
	// Deriving matrix C
	MatrixXd C = MatrixXd::Zero(6, 6);
	A(0, 0) = 1;
	A(1, 1) = 1;
	A(3, 0) = 1;
	A(4, 1) = 1;
	A(2, 3) = 1;
	A(3, 4) = 1;
	A(4, 5) = 1;
	// Inputvalues
	VectorXd x = VectorXd::Zero(11);
	x(0) = 0.3;
	x(1) = 0.4;
	x(2) = 0.1;
	x(3) = 0.05;
	x(4) = 0.01;
	x(5) = 0.005;
	x(6) = 0.5;
	x(7) = 1;
	x(8) = 0.001;
	x(9) = 0.002;
	x(10) = 0.01;
	VectorXd xl = x.block(0, 0, 6, 1);
	VectorXd xnl = x.block(6, 0, 5, 1);

	//std::cout << f(x) - (A*xl + f2(xnl)) << std::endl;

	std::vector<VectorXd> saved;
	long N = 1e7;
	auto start = std::chrono::system_clock::now();
	for (int n = 0; n < N; n++)
		saved.push_back(f(x));
	auto end = std::chrono::system_clock::now();
	long dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	printf("purely nonlinear function : %zd \n", dur / 1000);

	start = std::chrono::system_clock::now();
	for (int n = 0; n < N; n++)
		saved.push_back(A*xl + f2(xnl));
	end = std::chrono::system_clock::now();
	dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	printf("mixed function = %zd \n", dur / 1000);

	// covariance matrices
	MatrixXd S0 = MatrixXd::Zero(11,11);
	S0 << 0.0320, 0.0196, 0.0214, 0.0142, 0.0127, 0.0276, 0.0155, 0.0144, 0.0198, 0.0269, 0.0237,
		0.0196, 0.0481, 0.0237, 0.0256, 0.0189, 0.0369, 0.0197, 0.0235, 0.0249, 0.0338, 0.0300,
		0.0214, 0.0237, 0.0358, 0.0239, 0.0162, 0.0362, 0.0207, 0.0222, 0.0253, 0.0340, 0.0300,
		0.0142, 0.0256, 0.0239, 0.0668, 0.0154, 0.0323, 0.0250, 0.0234, 0.0285, 0.0294, 0.0245,
		0.0127, 0.0189, 0.0162, 0.0154, 0.0217, 0.0258, 0.0168, 0.0174, 0.0171, 0.0159, 0.0189,
		0.0276, 0.0369, 0.0362, 0.0323, 0.0258, 0.0570, 0.0290, 0.0322, 0.0327, 0.0412, 0.0360,
		0.0155, 0.0197, 0.0207, 0.0250, 0.0168, 0.0290, 0.0266, 0.0196, 0.0246, 0.0229, 0.0215,
		0.0144, 0.0235, 0.0222, 0.0234, 0.0174, 0.0322, 0.0196, 0.1889, 0.0215, 0.0316, 0.0265,
		0.0198, 0.0249, 0.0253, 0.0285, 0.0171, 0.0327, 0.0246, 0.0215, 0.0409, 0.0310, 0.0253,
		0.0269, 0.0338, 0.0340, 0.0294, 0.0159, 0.0412, 0.0229, 0.0316, 0.0310, 0.0657, 0.0446,
		0.0237, 0.0300, 0.0300, 0.0245, 0.0189, 0.0360, 0.0215, 0.0265, 0.0253, 0.0446, 0.0574;
	
	long N_UT = 1e6;
	{
		// Original SelUT
		VectorXd origninal_y;
		MatrixXd origninal_Sy, origninal_Syx;
		start = std::chrono::system_clock::now();
		for (long i = 0; i < N_UT; i++) {
			SelUT(x, S0, 11, f, origninal_y, origninal_Sy, origninal_Syx);
		}
		end = std::chrono::system_clock::now();
		dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		printf("original SelUT : %zd \n", dur / 1000);

		// Modified SelUT
		VectorXd modified_y;
		MatrixXd modified_Syx, modified_Sy;
		start = std::chrono::system_clock::now();
		for (long i = 0; i < N_UT; i++)
			SelUT(x, S0, 5, f2, A, modified_y, modified_Sy, modified_Syx);
		end = std::chrono::system_clock::now();
		dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		printf("modified SelUT : %zd \n", dur/1000);

		//std::cout << (origninal_y - modified_y).norm() / origninal_y.norm() << std::endl;

		//std::cout << (origninal_Sy - modified_Sy).norm() / origninal_Sy.norm() << std::endl;

		//std::cout << (origninal_Syx - modified_Syx).norm() / origninal_Syx.norm() << std::endl;

		//std::cout << std::endl << origninal_y << std::endl << std::endl << modified_y << std::endl;

		//std::cout << std::endl << origninal_Sy << std::endl << std::endl << modified_Sy << std::endl;

		//std::cout << std::endl << origninal_Syx << std::endl << std::endl << modified_Syx << std::endl;
	}
	{
		// Original UKF pred
		VectorXd original_x1, original_y;
		MatrixXd origninal_Sx1, origninal_Sx1x0, origninal_Sy, origninal_Syx1;
		start = std::chrono::system_clock::now();
		for (long i = 0; i < N_UT; i++) {
			SelUT(x, S0, 11, f, original_x1, origninal_Sx1, origninal_Sx1x0);
			SelUT(original_x1, origninal_Sx1, 11, g, original_y, origninal_Sy, origninal_Syx1);
		}
		end = std::chrono::system_clock::now();
		dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		printf("original UT2 : %zd \n", dur / 1000);

		// Modified UKF pred
		VectorXd modified_x1, modified_y;
		MatrixXd modified_Sx1, modified_Sx1x0, modified_Syx, modified_Sy;
		start = std::chrono::system_clock::now();
		for (long i = 0; i < N_UT; i++) {
			SelUT(x, S0, 5, f2, A, modified_x1, modified_Sx1, modified_Sx1x0);
			SelUT(modified_x1, modified_Sx1, 5, g2, C, modified_y, modified_Sy, modified_Syx);
		}
		end = std::chrono::system_clock::now();
		dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
		printf("modified UT2 : %zd \n", dur / 1000);
	}
	*/
	return 0;
}
