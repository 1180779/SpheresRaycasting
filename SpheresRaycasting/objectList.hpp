
#ifndef U1180779_OBJECTLIST_H
#define U1180779_OBJECTLIST_H

#include "shader.hpp"
#include "objectShape.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <string>

class objectList 
{
public:
    ~objectList();
    objectList& addObject(objectShape* obj) 
    {
        m_objs.push_back(obj);
        return *this;
    }
    void reset() { m_trans = glm::mat4(1.0f); }
    virtual objectList& translate(const glm::vec3& v)
    {
        m_trans = glm::translate(m_trans, v);
        return *this;
    }
    virtual objectList& rotate(const float angle, const glm::vec3& v)
    {
        m_trans = glm::rotate(m_trans, glm::radians(angle), glm::normalize(v));
        return *this;
    }
    virtual objectList& scale(const glm::vec3& v)
    {
        m_trans = glm::scale(m_trans, v);
        return *this;
    }

    void render(shader& sh, std::string name = "model");

    std::vector<objectShape*> m_objs;
    std::vector<glm::vec3> m_locations;
    glm::mat4 m_trans = glm::mat4(1.0f);
};

#endif