#pragma once

#include "readData.h"
#include "SLAMmodels.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include "UT.h"
#include "AUT.h"

using namespace Eigen;

struct SimSettings {
	bool enableKalmanFiltering = true, enableAddingLandmarks = true;
	double Sr = 1e-2;
	double Sphi = 5e-3;
	double Sv = 8e-2;
	double Somega = 10e-2;
	enum ComputationType { UT, NewUT } cType = UT;
	bool useAdaptive = false;
	std::string name;

	long time_start_sec, time_start_msec;
	double simulation_duration_sec;

	SimSettings(long time_start_sec, long time_start_msec, double duration_sec) :
		time_start_sec(time_start_sec), time_start_msec(time_start_msec), simulation_duration_sec(duration_sec) {};
};

class Simulator {
	SimSettings settings;

	// Codes to understand measurement data
	Barcodes barcodes;

	// Measurement data
	MeasurmentList measList;

	// Measured odometry data
	OdometryData odometryData;

	// Current odometry values
	double v, omega;

	// System states
	Eigen::VectorXd x;

	Eigen::MatrixXd Sx;

	std::vector<int> registeredLandmarks;

	// Clock [s]
	double t;

	// Objects to compute state and output update
	SLAMStateUpdate stateUpdater;
	SLAMOutputUpdate outputUpdater;

	// Logfile
	std::ofstream logfile;

	void _registerLandMark(int barcode, double R, double phi);
public:
	enum RunSetting { TIME_MEASURE, LOGPATH };

private:
	void _Step(double dT_sec, RunSetting set);

public:
	const Eigen::VectorXd& getx() const;

	const Eigen::MatrixXd& getSx() const;

	Simulator(SimSettings set) : barcodes(readBarcodes()),
		odometryData(ReadOdometryData(set.time_start_sec,set.time_start_msec,set.simulation_duration_sec)), settings(set){
		measList = ReadMeasurmentList(set.time_start_sec, set.time_start_msec, set.simulation_duration_sec,barcodes);
		std::cout << "Input files read (odometry: " << odometryData.size()
			<< ", measurements: " << measList.size() << ")" << std::endl;
		// create output file
		std::string filename = "Simulator_output_";
		filename += settings.name + "_";
		logfile.open(filename + std::to_string(std::chrono::system_clock::now().time_since_epoch().count())
			+ ".m", std::ios::out | std::ios::trunc);
	}

	~Simulator() {
		logfile.close();
	}

	void Run(RunSetting set) {
		int N = set == LOGPATH ? 1 : 20;
		auto start = std::chrono::system_clock::now();
		int Nmeas = 0;
		int maxmeas = 0;
		for (int n = 0; n < N; n++) {
			registeredLandmarks = {};
			v = 0;
			omega = 0;
			// Init car pose
			x = Eigen::VectorXd::Zero(3);
			x << 3.5732, -3.3328, 2.3408;
			Sx = Eigen::MatrixXd::Identity(3, 3)*0.001;
			// Init time
			t = 0;
			if (set == LOGPATH)
				logfile << "path_" << settings.name << "_t_x_y_phi = [ " << t << " " << x(0) << " " << x(1) << " " << x(2) << " ...\n";
			long next_meas = 0;
			long next_odometry = 0;
			while (next_meas < measList.size() || next_odometry < odometryData.size()) {
				// find next event
				bool measevent = next_meas < measList.size() &&
					(next_odometry >= odometryData.size() || measList[next_meas].t < odometryData[next_odometry].t);
				bool odometryevent = next_odometry < odometryData.size() && !measevent;
				if (measevent) { // measurement event
					// Check if next measevent shold be waited
					Nmeas++;
					long nextnext_meas = next_meas + 1;
					while (nextnext_meas < measList.size() && measList[nextnext_meas].t - measList[next_meas].t <= 0)
						nextnext_meas++;
					if (nextnext_meas - next_meas > maxmeas)
						maxmeas = nextnext_meas - next_meas;
					// STATE UPDATE
					// compute dt
					double dT = measList[nextnext_meas - 1].t - t;
					// time update
					_Step(dT, set);
					// OUTPUT UPDATE
					std::vector<int> actives; //measList indices
					std::vector<int> actives_x_indices; //x indices
					// gen list of re-recovered landmarks: actives, new ones simply add to the state machine
					for (long i = next_meas; i < nextnext_meas; i++) {
						// find  measList[i].barcode==registeredLandmarks[j]
						int j = -1;
						for (int k = 0; k < registeredLandmarks.size(); k++)
							if (registeredLandmarks[k] == measList[i].barcode) {
								j = k;
								break;
							}
						if (j == -1) {
							if (settings.enableAddingLandmarks)
								_registerLandMark(measList[i].barcode, measList[i].R, measList[i].phi); // if couldnt find
						}
						else { //if found save
							actives.push_back(i);
							actives_x_indices.push_back(j);
						}
					}
					if (actives.size() > 0 && settings.enableKalmanFiltering) {
						// Construct measurement vector
						Eigen::VectorXd y_meas(actives.size() * 2);
						for (int i = 0; i < actives.size(); i++) {
							y_meas(2 * i + 0) = measList[actives[i]].R;
							y_meas(2 * i + 1) = measList[actives[i]].phi;
						}
						// compute output for the actives
						Eigen::VectorXd y;
						Eigen::MatrixXd Sy, Sxy;
						if (settings.cType == SimSettings::NewUT && !settings.useAdaptive)
							outputUpdater.UT(actives_x_indices, x, Sx, y, Sy, Sxy);
						if (settings.cType == SimSettings::NewUT && settings.useAdaptive)
							outputUpdater.AUT(actives_x_indices, x, Sx, y_meas, y, Sy, Sxy);
						if (settings.cType == SimSettings::UT && !settings.useAdaptive)
							UT(x, Sx, SLAM_output_full_fcn(actives_x_indices), y, Sy, Sxy);
						if (settings.cType == SimSettings::UT && settings.useAdaptive)
							AUT(x, Sx, SLAM_output_full_fcn(actives_x_indices), y_meas, 0, 50, 0.5, y, Sy, Sxy);
						// Add measurement noise
						for (int i = 0; i < actives.size(); i++) {
							Sy(2 * i, 2 * i) += settings.Sr;
							Sy(2 * i + 1, 2 * i + 1) += settings.Sphi;
						}
						// Kalman
						Eigen::MatrixXd K = Sxy * Sy.inverse();
						Eigen::MatrixXd Sxnew = Sx - K * Sxy.transpose();
						Eigen::VectorXd ydiff = y_meas - y;
						for (int i = 0; i < actives.size(); i++) {
							ydiff(i) = fmod(ydiff(i), 2. * EIGEN_PI);
							if (ydiff(i) < -EIGEN_PI)
								ydiff(i) += EIGEN_PI * 2.;
							if (ydiff(i) > EIGEN_PI)
								ydiff(i) -= EIGEN_PI * 2.;
						}
						Eigen::VectorXd newx = x + K * ydiff;
						Eigen::MatrixXd newSx = (Sxnew + Sxnew.transpose()) / 2.;
						x = newx;
						Sx = newSx;
					}
					// clock update
					t = measList[next_meas].t;
					// index update
					next_meas = nextnext_meas;
				}
				if (odometryevent) { // odometry input
					// compute dt
					double dT = odometryData[next_odometry].t - t;
					// update odometry data
					v = odometryData[next_odometry].v;
					omega = odometryData[next_odometry].omega;
					// time update
					_Step(dT, set);
					// clock update
					t = odometryData[next_odometry].t;
					next_odometry++;
				}
			}
			if (set == LOGPATH) {
				logfile << "\t];\n";
				logfile << "x_" << settings.name << "=[ " << x.transpose() << "];\n";
				logfile << "Sx_" << settings.name << "=[ ";
				for (int i = 0; i < Sx.rows(); i++)
					logfile << "\t" << Sx.row(i) << ((i == Sx.rows() - 1) ? "];\n" : ";...\n");
				logfile << "landmarkIDs_" << settings.name << "=[ ";
				for (auto it : registeredLandmarks)
					logfile << it << " ";
				logfile << "];\n";
			}
		}
		auto end = std::chrono::system_clock::now();
		unsigned long dur = (unsigned long)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();	
		if (set == RunSetting::TIME_MEASURE) {
			std::cout << dur / N / 1000 << std::endl;
			logfile << "Tsim_ms_" << settings.name << " = " << dur / N / 1000 << ";\n";
		}
		std::cout << Nmeas << " " << maxmeas << std::endl;
	}
};