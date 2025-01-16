
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

    static const char* const typeString[];

    materialGenerator(type t) : kaGen(kaRange(t)), kdGen(kdRange(t)), ksGen(ksRange(t)), alphaGen(alphaRange(t)) { }

    inline material operator()() 
    {
        return material(kaGen(), ksGen(), kdGen(), alphaGen());
    }

private:
    range kaRange(type t);
    range kdRange(type t);
    range ksRange(type t);
    range alphaRange(type t);

    randomValueGenerator kdGen;
    randomValueGenerator ksGen;
    randomValueGenerator kaGen;
    randomValueGenerator alphaGen;
};

#endif