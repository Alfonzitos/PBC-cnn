#include "main.h"
#include "gatherData.h"
#include "error.h"
#include "network.h"
#include "layer.h"



volatile std::sig_atomic_t stopRequested = 0;

extern "C" void handleTerminationSignal(int signal)
{
    stopRequested = signal;
}

// application expects all image training data
// to have the same amount of rows and columns


int main(void)
{

    // std::string dataSetPath = "C:\\Users\\Alfons\\Downloads\\PBC_dataset_normal_DIB";
    std::string dataSetPath = "C:\\Users\\Alfons\\Desktop\\Biology\\ML\\PBC\\cropped";

    // move dataset path into conf file, make network read conffile, and pass requiredWidth/Height from conf
    //Data imageData(dataSetPath, 360, 363);

    if (std::signal(SIGINT, handleTerminationSignal) == SIG_ERR)
    {
        return -1;
    }
    if (std::signal(SIGTERM, handleTerminationSignal) == SIG_ERR)
    {
        return -1;
    }

    std::string networkConf = "src/network.json";
    Network net(networkConf);


    std::string testimg = "C:\\Users\\Alfons\\Desktop\\Biology\\ML\\PBC\\validation\\eosinophil\\EO_16412.jpg";

    net.classifyFile(testimg);
    //net.trainNetwork(imageData.data, 20, 32, networkConf, stopRequested);
    
    //net.saveNetwork(networkConf);

    return 0;
}
