# C++ Library for Relaxed Unscented Transformation

The package provides c++ implementation for computationally relaxed Unscented Transformation for related Kalman-filter applications


## Implemented methods ##

The c++ source of the implementated methods can be found in the folder "*cppsource*".


-  It depends only on Eigen library (used version 3.3.7).

-  The header RelaxedUT.h containts the new implementated methods and the header UT.h contains a simple implementation of the original Unscented transformation for the sake of comparisons

- The relaxed method can decrease the computational cost to up to 30%  for **partially linear, larger dynamical models**.

For more details, see papers:

- J. Kuti and P. Galambos, “Decreasing the Computational Demand of
Unscented Kalman Filter based Methods,” in Proc. Of IEEE 15th
International Symposium on Applied Computational Intelligence and
Informatics,(SACI 2021), accepted.

- J. Kuti and P. Galambos,, “Computational Analysis of Relaxed Unscented Transformation
in terms of necessary floating point operations,” in Proc. Of IEEE
25th IEEE International Conference on Intelligent Engineering Systems
2021,(INES 2021), submitted.

- S. J. Julier and J. K. Uhlmann, “Unscented filtering and nonlinear
estimation,” Proceedings of the IEEE, vol. 92, no. 3, pp. 401–422, 2004

and the included documentation.

## Examples ##

In the folder "cpptester", examples show the precision, usage and benefits of the methods. The provided CMake script is able to initialize the VisualStudio project/Makefile/etc. environment. Then the implemented cases can be checked/modified/etc.

(Tested on Win10, Visual Studio 2017 x64, it also needs an installed Eigen3 package)

Steps of intall:

- CMake: configure (set Eigen path, folder for binaries, compiler), then generate

- build/modify the examples, enjoy and understand