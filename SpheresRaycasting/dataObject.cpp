
#include "dataObject.hpp"
#include "cudaWrappers.hpp"
#include <iostream>

void dataObject::generate(
    unsigned int count, 
    range rR, range xR, 
    range yR, range zR, 
    materialGenerator::type t)
{
    randomValueGenerator rGen(rR);
    randomValueGenerator xGen(xR);
    randomValueGenerator yGen(yR);
    randomValueGenerator zGen(zR);
    randomValueGenerator colorGen(0.0f, 1.0f);

    materialGenerator matGen(t);

    m_objs.reserve(count);
    for (int i = 0; i < count; ++i) {
        // TODO: check that spheres do not overlap
        unifiedObject obj;
        obj.x = xGen();
        obj.y = yGen();
        obj.z = zGen();
        obj.w = 1.0f;
        obj.r = rGen();
        

        obj.type = types::sphere;

        obj.color.x = colorGen();
        obj.color.y = colorGen();
        obj.color.z = colorGen();

        material m = matGen();

        obj.ka = m.ka;
        obj.ks = m.ks;
        obj.kd = m.kd;
        obj.alpha = m.alpha;

        assert(obj.alpha > 0.999f);
        assert(obj.ka >= 0.0f && obj.ka <= 1.0f);

        m_objs.push_back(obj);
        //std::cout << "x = " << obj.x << ", y = " << obj.y << ", z = " << obj.z << ", r = " << obj.r << std::endl;
    }
}

void dataObject::generateLights(
    unsigned int count,
    range rR, range xR,
    range yR, range zR)
{
    randomValueGenerator rGen(rR);
    randomValueGenerator xGen(xR);
    randomValueGenerator yGen(yR);
    randomValueGenerator zGen(zR);
    randomValueGenerator colorGen(255.0f / 2.0f);

    mh_lights.hMalloc(count);
    for (int i = 0; i < count; ++i) {
        // TODO: check that spheres do not overlap
        unifiedObject obj;
        obj.x = xGen();
        obj.y = yGen();
        obj.z = zGen();
        obj.w = 1.0f;
        obj.r = rGen();

        obj.type = types::lightSource;

        // not used yet
        obj.color.x = 1.0f;
        obj.color.y = 1.0f;
        obj.color.z = 1.0f;

        obj.ka = 0.5f;
        obj.kd = 0.5f;
        obj.ks = 0.5f;
        obj.alpha = 1.0f;

        // save generated data
        mh_lights.x[i] = obj.x;
        mh_lights.y[i] = obj.y;
        mh_lights.z[i] = obj.z;
        mh_lights.w[i] = obj.w;

        mh_lights.is[i] = 0.2f;
        mh_lights.id[i] = 0.2f;

        m_objs.push_back(obj);
        //std::cout << "x = " << obj.x << ", y = " << obj.y << ", z = " << obj.z << ", r = " << obj.r << std::endl;
    }
    md_lights.dMalloc(count);
    md_lights.copyToDeviceFromHost(mh_lights);
}

void dataObject::freeLights()
{
    md_lights.dFree();
    mh_lights.hFree();
}


