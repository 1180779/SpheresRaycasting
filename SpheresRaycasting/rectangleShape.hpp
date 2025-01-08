
#ifndef U1180779_RECTANGLE_H
#define U1180779_RECTANGLE_H

#include "objectShape.hpp"

class rectangle : public objectShape
{
public:
    rectangle();
    ~rectangle();
    virtual void render() const;

private:
    GLuint m_VBO[3];
    GLuint m_EBO;
    GLuint m_VAO;

    float m_vertices[12] = {
         0.5f,  0.5f, 0.0f,     // top right
         0.5f, -0.5f, 0.0f,     // bottom right
        -0.5f, -0.5f, 0.0f,     // bottom left
        -0.5f,  0.5f, 0.0f      // top left 
    };

    float m_colors[12] = { // colors           
        1.0f, 0.0f, 0.0f,       // top right
        0.0f, 1.0f, 0.0f,       // bottom right
        0.0f, 0.0f, 1.0f,       // bottom left
        1.0f, 1.0f, 0.0f,       // top left };
    };
    float m_texture[8] = { // texture coords
        1.0f, 1.0f,             // top right
        1.0f, 0.0f,             // bottom right
        0.0f, 0.0f,             // bottom left
        0.0f, 1.0f              // top left };
    };

    unsigned int m_indices[6] = {  // note that we start from 0!
        0, 1, 3,   // first triangle
        1, 2, 3    // second triangle
    };
};

#endif