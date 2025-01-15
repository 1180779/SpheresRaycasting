
#ifndef U1180779_RANDOM_VALUE_GENERATOR_H
#define U1180779_RANDOM_VALUE_GENERATOR_H

struct range
{
    range() : min(0), max(0) { }
    range(float min, float max) : min(min), max(max) { }

    float min;
    float max;
};

struct randomValueGenerator
{
    randomValueGenerator(range r) : r(r)
    {
        srand(time(0));
    }

    randomValueGenerator(float min, float max) : r(min, max)
    {
        srand(time(0));
    }

    randomValueGenerator(float f) : r(f, f)
    {
        srand(time(0));
    }

    const range r;

    inline float operator()()
    {
        return r.min + static_cast<float>(
            rand()) / (static_cast<float> (RAND_MAX / (r.max - r.min)));
    }
};

#endif
