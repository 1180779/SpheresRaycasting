#include "objectList.hpp"
#include <glm/gtc/matrix_transform.hpp>

objectList::~objectList()
{
    for (auto o : m_objs) {
        delete o;
    }
}

void objectList::render(shader& sh, std::string modelName)
{
    for (int i = 0; i < m_objs.size(); ++i) {
        sh.setMatrix4fv(modelName, m_trans * m_objs[i]->model());
        //sh.setMatrix4fv(name, glm::translate(
        //    glm::rotate(m_objs[i]->model(), glm::radians(m_angle), m_v), m_locations[i]));
        m_objs[i]->render();
    }
}
