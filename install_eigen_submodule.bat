git submodule init
git submodule update
IF "%2"=="" cmake eigen -Seigen -B"%1" -DBUILD_TESTING=FALSE -DCMAKE_INSTALL_INCLUDEDIR="include"
ELSE cmake eigen -Seigen -B"%1" -DBUILD_TESTING=FALSE -DCMAKE_INSTALL_PREFIX="%2" -DCMAKE_INSTALL_INCLUDEDIR="include"
cmake --install "%1" --config Release

