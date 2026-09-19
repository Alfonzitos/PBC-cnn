#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <cmath>
#include <random>
#include <filesystem>

#include "../3DPartyLibs/json.hpp"
using json = nlohmann::json;

enum Answers
{
    basophil,
    eosinophil,
    erythroblast,
    ig,
    lymphocyte,
    monocyte,
    neutrophil,
    platelet
};

class Tensor
{
public:
    // https://stackoverflow.com/questions/24285112/why-must-initializer-list-order-match-member-declaration-order
    int channels = 0;
    int width = 0;
    int height = 0;
    Eigen::VectorXf data;

    Tensor() = default;

    Tensor(int channels, int width, int height);

    Tensor(Eigen::VectorXf vector);
    void resize(int newChannels, int newWidth, int newHeight);

    int index(int channel, int x, int y);

    float &at(int channel, int x, int y);
};

class Kernel
{

public:
    Tensor weights;
    float bias = 0.0f;

    Kernel(int channels, int width, int height);

    float &at(int channel, int x, int y);
};

Eigen::VectorXf getAnswerVector(std::string dirName);

void createParentDirectory(std::string &path);
void saveMatrix(std::string &path, Eigen::MatrixXf &matrix);
void loadMatrix(std::string &path, Eigen::MatrixXf &matrix);
void saveVector(std::string &path, Eigen::VectorXf &vector);
void loadVector(std::string &path, Eigen::VectorXf &vector);
void saveFilters(std::string &path, std::vector<Kernel> &filters);
void loadFilters(std::string &path, std::vector<Kernel> &filters);
void saveFilterBiases(std::string &path, std::vector<Kernel> &filters);
void loadFilterBiases(std::string &path, std::vector<Kernel> &filters);

Eigen::MatrixXf createRandomWeights(int outputCount, int inputCount);
