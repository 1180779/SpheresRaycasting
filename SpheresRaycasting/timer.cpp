
#include "timer.hpp"

void timer::start()
{
    m_time = std::chrono::high_resolution_clock::now();
}

void timer::stop(std::string functionName)
{
    auto stop = std::chrono::high_resolution_clock::now();
    auto microSec = std::chrono::duration_cast<std::chrono::microseconds>(stop - m_time);
    std::cout << "function " << functionName << " took: " << microSec.count() / 1000 << " miliseconds, " << microSec.count() % 1000 << " microseconds" << std::endl;
}
