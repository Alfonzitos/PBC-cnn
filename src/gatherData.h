#pragma once

#include <string>
#include <vector>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#include "helpFuncs.h"

Tensor matToTensor(const cv::Mat &image);

enum pathInfo
{
    IMAGE,
    DIR,
    INVALID
};

struct trainingInstance
{
    std::string dataType;
    Tensor data;
};

class Data
{
public:
    // dataFolderPath must be a folder containing ONLY subfolders where each subfolder
    //  contains a single "type" of training data(1 output neuron), such as images
    Data(std::string &dataFolderPath, int requiredWidth, int requiredHeight);

    std::vector<trainingInstance> data;

    static cv::Mat readFile(std::string &filePath, int requiredWidth, int requiredHeight);

private:
    pathInfo interrogatePath(std::string &path);
    std::vector<std::string> getSubDirNames(std::string &path);
    std::vector<std::string> getFilePaths(std::string &path);
};
