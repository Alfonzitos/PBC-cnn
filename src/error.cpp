#include "error.h"

Error::Error(std::string err)
{
    std::cerr << "\033[41;196mERROR OCCURED: " << err << "\033[m" << std::endl;
    exit(-1);
}