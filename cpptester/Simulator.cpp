#include "Simulator.h"
#include <algorithm>
#include <chrono>
#include "UT.h"
#include "AUT.h"
using namespace Eigen;
using namespace RelaxedUT;

Eigen::VectorXd computeNewLandmarkPosition(const Eigen::VectorXd & rdelta_x) {
	Eigen::VectorXd out(2);
	out(0) = rdelta_x(2) + rdelta_x(0)*cos(rdelta_x(1) + rdelta_x(4));
	out(1) = rdelta_x(3) + rdelta_x(0)*sin(rdelta_x(1) + rdelta_x(4));
	return out;
}

void Simulator::_registerLandMark(int barcode, double R, double phi) {
	// Add initial pose (computing its variance and covariances via UT)
	Eigen::VectorXd xy_lm;
	Eigen::MatrixXd Sxylm, Sx_xylm;
	Eigen::VectorXd in(x.size() + 2);
	in(0) = R;
	in(1) = phi;
	in.segment(2, x.size()) = x;
	Eigen::MatrixXd Sin = Eigen::MatrixXd::Zero(x.size() + 2, x.size() + 2);
	Sin(0, 0) = settings.Sr;
	Sin(1, 1) = settings.Sphi;
	Sin.block(2, 2, x.size(), x.size()) = Sx;
	UT(in, Sin, computeNewLandmarkPosition, xy_lm, Sxylm, Sx_xylm);
	Eigen::VectorXd newx(x.size() + 2);
	newx.segment(0, x.size()) = x;
	newx.segment(x.size(), 2) = xy_lm;
	Eigen::MatrixXd newSx = Eigen::MatrixXd::Zero(x.size() + 2, x.size() + 2);
	newSx.block(0, 0, x.size(), x.size()) = Sx;
	newSx.block(0, x.size(), x.size(), 2) = Sx_xylm.block(2, 0, x.size(), 2);
	newSx.block(x.size(), 0, 2, x.size()) = Sx_xylm.block(2, 0, x.size(), 2).transpose();
	newSx.block(x.size(), x.size(), 2, 2) = Sxylm;
	x = newx;
	Sx = newSx;
	registeredLandmarks.push_back(barcode);
}

void Simulator::_Step(double dT_sec, RunSetting set) {
	Eigen::VectorXd y;
	Eigen::MatrixXd Sy, Syx0;
	if (settings.cType == SimSettings::NewUT) {
		stateUpdater.UT(dT_sec, v, omega, settings.Sv, settings.Somega, x, Sx, y, Sy, Syx0);
		x = y;
		Sx = Sy;
	}
	else {
		Eigen::VectorXd in(x.size() + 2);
		in(0) = v;
		in(1) = omega;
		in.segment(2, x.size()) = x;
		Eigen::MatrixXd Sin = Eigen::MatrixXd::Zero(x.size() + 2, x.size() + 2);
		Sin(0, 0) = settings.Sv;
		Sin(1, 1) = settings.Somega;
		Sin.block(2, 2, x.size(), x.size()) = Sx;
		UT(in, Sin, [dT_sec](const Eigen::VectorXd& in)->Eigen::VectorXd {return SLAMStateUpdateFull(in, dT_sec); }, y, Sy, Syx0);
		x = y;
		Sx = Sy;
	}
	if (set == LOGPATH)
		logfile << "\t; " << t << " " << x(0) << " " << x(1) << " " << x(2) << " ...\n";
}

const Eigen::VectorXd & Simulator::getx() const {
	return x;
}

const Eigen::MatrixXd & Simulator::getSx() const {
	return Sx;
}

Simulator::Simulator(SimSettings set) : barcodes(readBarcodes()),
odometryData(ReadOdometryData(set.time_start_sec, set.time_start_msec, set.simulation_duration_sec)), settings(set) {
	measList = ReadMeasurmentList(set.time_start_sec, set.time_start_msec, set.simulation_duration_sec, barcodes);
	// create output file
	std::string filename = std::string(DATASET_PATH) + "Simulator_output_";
	if (settings.cType == settings.UT)
		filename += "UT";
	else
		filename += "RelaxedUT";
	logfile.open(filename + ".m", std::ios::out | std::ios::trunc);
}

Simulator::~Simulator() {
	logfile.close();
}

void Simulator::Run(RunSetting set) {
	simResults = SimResults();
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
			logfile << "path_" << "_t_x_y_phi = [ " << t << " " << x(0) << " " << x(1) << " " << x(2) << " ...\n";
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
					for (int i = 0; i < actives.size(); i++) { // offset angle differences into (-pi,pi]
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
			logfile << "x_" << "=[ " << x.transpose() << "];\n";
			logfile << "Sx_" << "=[ ";
			for (int i = 0; i < Sx.rows(); i++)
				logfile << "\t" << Sx.row(i) << ((i == Sx.rows() - 1) ? "];\n" : ";...\n");
			logfile << "landmarkIDs_" << "=[ ";
			for (auto it : registeredLandmarks)
				logfile << it << " ";
			logfile << "];\n";
		}
	}
	auto end = std::chrono::system_clock::now();
	unsigned long dur = (unsigned long)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	if (set == RunSetting::TIME_MEASURE) {
		simResults.Nsteps = N;
		simResults.Nmeas = Nmeas;
		simResults.duration_sec = dur / 1e6;
		simResults.average_duration_step = simResults.duration_sec / N;
		logfile << "Tsim_ms_" << " = " << dur / N / 1000 << ";\n";
	}
}

Simulator::SimResults Simulator::getSimResults() const {
	return simResults;
}

SimSettings::SimSettings(long time_start_sec, long time_start_msec, double duration_sec) :
	time_start_sec(time_start_sec), time_start_msec(time_start_msec), simulation_duration_sec(duration_sec) {}
