
#ifndef U1180779_SCENE_CONFIG_H
#define U1180779_SCENE_CONFIG_H

#include "range.hpp"
#include "materialGenerator.hpp"

struct sceneConfig 
{
    unsigned int sCount;
    range sXR;
    range sYR;
    range sZR;
    range sRR;
    
    unsigned int lCount;
    range lXR;
    range lYR;
    range lZR;
    range lRR;

    materialGenerator::type matType;
};

#endif