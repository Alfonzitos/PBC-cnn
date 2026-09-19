#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <algorithm>
#include <random>
#include <csignal>
#include <chrono>
#include <array>
#include <iomanip>

#include "layer.h"
#include "helpFuncs.h"
#include "gatherData.h"

constexpr std::size_t CLASS_COUNT = 8;

struct EvaluationResult
{
    float averageCost = 0.0f;
    float accuracy = 0.0f;

    std::size_t correctPredictions = 0;
    std::size_t sampleCount = 0;

    // rows = actual classes
    // columns = predicted classes
    std::array<std::array<std::size_t, CLASS_COUNT>, CLASS_COUNT>confusion{};
};

class Network
{
public:
    Network(std::string &networkConfig);
    void trainNetwork(std::vector<trainingInstance> &trainingData, int trainingRuns,
                      int batchSize, std::string configPath,
                      volatile std::sig_atomic_t &stopRequested);
    void saveNetwork(std::string &configPath);
    EvaluationResult evaluateDataset(std::vector<trainingInstance> &dataset);
    Tensor &classifyImage(Tensor &inputVector);
    void classifyFile(const std::string &imagePath);

private:
    void backPropagate(Tensor &previousActivation, Tensor &correctAnswer);

    std::vector<std::unique_ptr<Layer>> layers;

    float cost(Tensor &outputVector, Tensor &answerVector);
};