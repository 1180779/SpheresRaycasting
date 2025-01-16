
#ifndef U1180779_RANGE_H
#define U1180779_RANGE_H

struct range
{
    range() : min(0), max(0) { }
    range(float min, float max) : min(min), max(max) { }

    float min;
    float max;

    inline float avg() { return (min + max) / 2.0f; }
};

#endif