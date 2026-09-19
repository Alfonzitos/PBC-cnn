cmake_minimum_required(VERSION 3.16)

set(
    CMAKE_TOOLCHAIN_FILE
    "C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING ""
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(MyProject LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(app
    src/main.cpp
    src/gatherData.cpp
    src/error.cpp
    src/network.cpp
    src/layer.cpp
    src/helpFuncs.cpp
    src/helpMath.cpp
)

target_compile_features(app PRIVATE cxx_std_17)

target_link_libraries(app PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)