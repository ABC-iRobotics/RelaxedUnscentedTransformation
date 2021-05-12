git submodule init
git submodule update
set build_path=%1
set install_path=%2
cmake eigen -Seigen -B"%build_path%/Third party/eigen" -DBUILD_TESTING=FALSE -DCMAKE_INSTALL_PREFIX="%install_path%/Third party/eigen" -DCMAKE_INSTALL_INCLUDEDIR="include"
cmake --install "%build_path%/Third party/eigen" --config Release

