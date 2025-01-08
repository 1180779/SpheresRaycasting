
#ifndef U1180779_TRIANGLE_H
#define U1180779_TRIANGLE_H

#include "objectShape.hpp"

class triangleShape : public objectShape
{
public:
    triangleShape();
    virtual void render() const;

private:
    GLuint m_VBO;
    GLuint m_VAO;

    float m_vertices[9] = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f
    };
};


#endif