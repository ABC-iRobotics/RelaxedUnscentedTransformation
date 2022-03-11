#include <iostream>
#include "Simulator.h"

// This example shows how a more complex problem can be handled via the library
// Simulations can be performed on one of the "UTIAS Multi-Robot Cooperative Localization and Mapping Dataset"
// with or without registering the seen landmarks / performing Kalman-filtering using an adaptive extension of
// Unscented Kalman-fitering
//
// The simulation is performed for even UT and relaxed UT twice:
//  - First time, the path and final state is saved into
//    a log file, that can be visualized via the plotter.m file.
//  - Second time, the computation time is measured and comparised.
//
int main(void) {
	auto Norder = { 2,3,4 };

	// Init measurement data
	long time_start_sec = 1248272272, time_start_msec = 841;
	double simulation_duration_sec = 80;
	SimSettings set(time_start_sec, time_start_msec, simulation_duration_sec);
	ConvertGroundTruthData(time_start_sec, time_start_msec, simulation_duration_sec);

	// Perform simulations for UT and relaxed UT cases
	for (auto NorderIt : Norder)
	{
		set.Norder = NorderIt;
		set.enableAddingLandmarks = true;
		set.enableKalmanFiltering = true;
		set.useAdaptive = false;
		for (int i = 0; i < 2; i++) {
			if (i == 0) {
				set.name = "UT_" + std::to_string(set.Norder);
				set.cType = SimSettings::UT;
				std::cout << "Traditional UT:\n";
			}
			else {
				set.name = "relaxedUT_" + std::to_string(set.Norder);
				set.cType = SimSettings::NewUT;
				std::cout << "Relaxed UT:\n";
			}
			Simulator sim(set);
			sim.Run(Simulator::LOGPATH);
			sim.Run(Simulator::TIME_MEASURE);
			auto res = sim.getSimResults();
			std::cout << " Average step time: " << res.average_duration_step << "[s]\n Total time: " <<
				res.duration_sec << "[s]" << std::endl;
		}

		std::cout << "The logs can be visualized via the plotter.m matlab script.\n";
	}
	return 0;
}
