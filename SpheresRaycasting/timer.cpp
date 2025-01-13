
#include "timer.hpp"

void timer::start()
{
    m_time = std::chrono::high_resolution_clock::now();
}

void timer::stop(std::string functionName)
{
    auto stop = std::chrono::high_resolution_clock::now();
    auto microSec = std::chrono::duration_cast<std::chrono::milliseconds>(stop - m_time);
    std::cout << "function " << functionName << " took: " << microSec.count() << "miliseconds" << std::endl;
}
