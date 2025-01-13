
#ifndef U1180779_TIMER_H
#define U1180779_TIMER_H

#include <string>
#include <chrono>
#include <iostream>

class timer
{
public:
    void start();
    void stop(std::string functionName);

private:
    std::chrono::steady_clock::time_point m_time;
};

#endif