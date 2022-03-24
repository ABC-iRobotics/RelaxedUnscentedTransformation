#pragma once
#include "readData.h"
#include "SLAMmodels.h"
#include <fstream>

struct SimSettings {
	bool enableAddingLandmarks = true; // if barcodes are added from the measurements
	bool enableKalmanFiltering = true; // if Kalman-filtering is applied to correct the barcode positions
	
	double Sr = 10e-2; // variance of measured distance
	double Sphi = 5e-3; // variance of measured angle
	double Sv = 8e-2; // variance of velocity comes from odometry
	double Somega = 20e-2; // variance of angular velocity comes from odometry
	enum ComputationType { UT, NewUT } cType = UT; // if original or relaxed UT will be applied
	bool useAdaptive = false; // if adaptive method should be used
	std::string name; // name of the settings

	long time_start_sec, time_start_msec; // start of the log should be simulated
	double simulation_duration_sec; // duration of the simulation
	int Norder;

	SimSettings(long time_start_sec, long time_start_msec, double duration_sec); // Constructor
};

class Simulator {
	// Settings set at initialization
	SimSettings settings;

	// Codes to understand measurement data
	Barcodes barcodes;

	// Measurement data
	MeasurmentList measList;

	// Measured odometry data
	OdometryData odometryData;

	// Current odometry values
	double v, omega;

	// Estimated system states
	Eigen::VectorXd x;

	// Estimated covariances
	Eigen::MatrixXd Sx;

	// Set and order of registered landmarks
	std::vector<int> registeredLandmarks;

	// Clock [s]
	double t;

	// Objects to compute state and output update
	SLAMStateUpdate stateUpdater;
	SLAMOutputUpdate outputUpdater;

	// Logfile
	std::ofstream logfile;

	// Called if new landmark was seen
	void _registerLandMark(int barcode, double R, double phi, int Norder);

	struct SimResults {
		long Nsteps, Nmeas;
		double duration_sec;
		double average_duration_step;
		SimResults() : Nsteps(NAN), Nmeas(NAN), duration_sec(NAN), average_duration_step(NAN) {};
	} simResults;

public:
	enum RunSetting { TIME_MEASURE, LOGPATH }; // You can save the computed path or measure computation time (together it is not precise)

private:
	// Prediciton for "dT_sec" time with or without logging it into file according to the RunSetting
	void _Step(double dT_sec, RunSetting set);

public:
	const Eigen::VectorXd& getx() const; // Returns estimated state

	const Eigen::MatrixXd& getSx() const; // Returns estimated state covariance matrix

	Simulator(SimSettings set); // Constructor

	~Simulator(); // Denstructor

	void Run(RunSetting set); // Performing the simulation with or without logging

	SimResults getSimResults() const;
};