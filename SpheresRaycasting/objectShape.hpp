
#ifndef U1180779_OBJECT_H
#define U1180779_OBJECT_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class objectShape
{
public:
    virtual void render() const = 0;
    virtual glm::mat4 model() { return m_modelTrans; }
    virtual void reset() { m_modelTrans = m_model; }
    virtual objectShape& translate(const glm::vec3& v) 
    {
        m_modelTrans = glm::translate(m_modelTrans, v); 
        return *this;
    }
    virtual objectShape& rotate(const float angle, const glm::vec3& v)
    {
        m_modelTrans = glm::rotate(m_modelTrans, glm::radians(angle), v); 
        return *this;
    }
    virtual objectShape& scale(const glm::vec3& v)
    {
        m_modelTrans = glm::scale(m_modelTrans, v);
        return *this;
    }

private:
    const static glm::mat4 m_model;
    glm::mat4 m_modelTrans = glm::mat4(1.0f);
};

#endif