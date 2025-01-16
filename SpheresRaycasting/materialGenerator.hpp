
#ifndef U1180779_RANDOM_METERIAL_GENERATOR_H
#define U1180779_RANDOM_METERIAL_GENERATOR_H

#include "randomValueGenerator.hpp"

struct material 
{
    material() : ka(0), ks(0), kd(0), alpha(0) { }
    material(float ka, float ks, float kd, float alpha) 
        : ka(ka), ks(ks), kd(kd), alpha(alpha) { }
    
    float ka; /* [0, 1] */
    float ks; /* [0, 1] */
    float kd; /* [0, 1] */
    float alpha; /* >= 1 */
};

class materialGenerator 
{
public:
    enum type
    {
        matte,
        plastic,
        glossy, 
        metallic,
        reflective,
        soft,
        selfLuminous
    };

    materialGenerator(type t) : kaGen(kaRange(t)), kdGen(kdRange(t)), ksGen(ksRange(t)), alphaGen(alphaRange(t)) { }

    inline material operator()() 
    {
        return material(kaGen(), ksGen(), kdGen(), alphaGen());
    }

private:
    range kaRange(type t) 
    {
        switch(t) 
        {
        case matte:         return range(0.2f, 0.4f);
        case plastic:       return range(0.3f, 0.5f);
        case glossy:        return range(0.2f, 0.5f);
        case metallic:      return range(0.1f, 0.3f);
        case reflective:    return range(0.0f, 0.2f);
        case soft:          return range(0.3f, 0.6f);
        case selfLuminous:  return range(0.8f, 1.0f);
        }
        return range(0.0f, 1.0f);
    }

    range kdRange(type t)
    {
        switch (t)
        {
        case matte:         return range(0.7f, 1.0f);
        case plastic:       return range(0.5f, 0.7f);
        case glossy:        return range(0.4f, 0.6f);
        case metallic:      return range(0.0f, 0.3f);
        case reflective:    return range(0.0f, 0.2f);
        case soft:          return range(0.6f, 0.9f);
        case selfLuminous:  return range(0.3f, 0.6f);
        }
        return range(0.0f, 1.0f);
    }

    range ksRange(type t)
    {
        switch (t)
        {
        case matte:         return range(0.0f, 0.2f);
        case plastic:       return range(0.2f, 0.4f);
        case glossy:        return range(0.5f, 0.7f);
        case metallic:      return range(0.7f, 1.0f);
        case reflective:    return range(0.8f, 1.0f);
        case soft:          return range(0.1f, 0.3f);
        case selfLuminous:  return range(0.2f, 0.4f);
        }
        return range(0.0f, 1.0f);
    }

    range alphaRange(type t) 
    {
        switch(t) 
        {
        case matte:         return range(1.0f, 5.0f);
        case plastic:       return range(10.0f, 20.0f);
        case glossy:        return range(20.0f, 50.0f);
        case metallic:      return range(50.0f, 200.0f);
        case reflective:    return range(100.0f, 500.0f);
        case soft:          return range(5.0f, 15.0f);
        case selfLuminous:  return range(10.0f, 30.0f);
        }
        return range(1.0f, 2.0f);
    }

    randomValueGenerator kdGen;
    randomValueGenerator ksGen;
    randomValueGenerator kaGen;
    randomValueGenerator alphaGen;
};

#endif