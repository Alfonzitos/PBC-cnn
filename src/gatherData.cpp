#include "gatherData.h"
#include "error.h"

Data::Data(std::string &path, int requiredWidth, int requiredHeight)
{
    pathInfo pathType = interrogatePath(path);

    std::vector<trainingInstance> actualData;

    std::vector<std::string> subDirs = getSubDirNames(path);
    if (pathType == DIR)
    {
        for (std::string &subDirName : subDirs)
        {
            int temp = 0;
            std::cout << path + "\\" + subDirName << std::endl;

            std::vector<std::string> filePaths = getFilePaths(path + "\\" + subDirName);

            actualData.reserve(actualData.size() + filePaths.size());

            for (std::string &file : filePaths)
            {
                // if (temp >= 1)
                //     break;
                // temp++;
                trainingInstance instance;

                actualData.push_back({subDirName, matToTensor(readFile(file, requiredWidth, requiredHeight))});
            }
        }

        data = std::move(actualData);
    }
    else if (pathType == IMAGE)
    {
        Error("Path supplied is IMAGE, not folder");
    }
    else if (pathType == INVALID)
    {
        Error("Invalid path");
    }
}

cv::Mat Data::readFile(std::string &filePath, int requiredWidth, int requiredHeight)
{

    cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR_RGB);

    if (image.cols != requiredWidth || image.rows != requiredHeight)
    {
        cv::resize(image, image, cv::Size(requiredWidth, requiredHeight), 0.0, 0.0, cv::INTER_AREA);
    
    }
    if (image.empty())
    {
        Error("Image not able to be read");
    }
    return image;
}

std::vector<std::string> Data::getSubDirNames(std::string& path)
{
    std::vector<std::string> subDirectories;
    for (const auto &entry : std::filesystem::directory_iterator(path))
    {
        std::string currentPath = entry.path().generic_string();
        pathInfo pathType = interrogatePath(currentPath);

        if (pathType == IMAGE || pathType == INVALID)
        {
            Error("Top level directory path supplied must only contain folders");
        }
        else if (pathType == DIR)
        {
            subDirectories.push_back(entry.path().filename().string());
        }
    }
    if (subDirectories.size() == 0)
    {
        Error("Top level direcotry path supplied is empty");
    }

    return subDirectories;
}

std::vector<std::string> Data::getFilePaths(std::string& path)
{
    std::vector<std::string> files;
    for (const auto &entry : std::filesystem::directory_iterator(path))
    {
        std::string currentPath = entry.path().generic_string();
        pathInfo pathType = interrogatePath(currentPath);

        if (pathType == DIR || pathType == INVALID)
        {
            Error("Subdirectory must only contain images");
        }
        else if (pathType == IMAGE)
        {
            files.push_back(currentPath);
        }
    }
    if (files.size() == 0)
    {
        Error("All subdirectories must contain atleast one image");
    }
    return files;
}

pathInfo Data::interrogatePath(std::string &path)
{
    struct stat s{0};

    if (stat(path.c_str(), &s) == 0)
    {
        if (std::filesystem::is_directory(path))
        {
            return DIR;
        }
        else if (std::filesystem::is_regular_file(path))
        {
            return IMAGE;
        }
        else
        {
            return INVALID;
        }
    }
    else
    {
        return INVALID;
    }
}

Tensor matToTensor(const cv::Mat &image)
{
    int channels = image.channels();
    int width = image.cols;
    int height = image.rows;

    Tensor tensor(channels, width, height);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int c = 0; c < channels; c++)
            {
                tensor.at(c, x, y) =
                    static_cast<float>(image.ptr<uchar>(y)[x * channels + c]) / 255;
            }
        }
    }

    return tensor;
}