
#include "sceneConfig.hpp"

static void eraseWhite(std::string& str);
static std::string nextstrToSpace(std::string& str);

void sceneConfig::loadFromFile(const char* filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Could not open configuration file (\'" << filename << "\')" << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.size() == 0)
            continue;
        if (line[0] == '#')
            continue;

        std::string option = nextstrToSpace(line);
        /* sphere fields */
        if (option == "sCount") {
            sCount = std::stoi(line);
        }

        else if (option == "sXRmin") {
            sXR.min = std::stof(line);
        }
        else if (option == "sXRmax") {
            sXR.max = std::stof(line);
        }

        else if (option == "sYRmin") {
            sYR.min = std::stof(line);
        }
        else if (option == "sYRmax") {
            sYR.max = std::stof(line);
        }

        else if (option == "sZRmin") {
            sZR.min = std::stof(line);
        }
        else if (option == "sZRmax") {
            sZR.max = std::stof(line);
        }

        else if (option == "sRRmin") {
            sRR.min = std::stof(line);
        }
        else if (option == "sRRmax") {
            sRR.max = std::stof(line);
        }

        /* light fields */

        if (option == "lCount") {
            lCount = std::stoi(line);
        }

        else if (option == "lXRmin") {
            lXR.min = std::stof(line);
        }
        else if (option == "lXRmax") {
            lXR.max = std::stof(line);
        }

        else if (option == "lYRmin") {
            lYR.min = std::stof(line);
        }
        else if (option == "lYRmax") {
            lYR.max = std::stof(line);
        }

        else if (option == "lZRmin") {
            lZR.min = std::stof(line);
        }
        else if (option == "lZRmax") {
            lZR.max = std::stof(line);
        }

        else if (option == "lRRmin") {
            lRR.min = std::stof(line);
        }
        else if (option == "lRRmax") {
            lRR.max = std::stof(line);
        }

        else if (option == "isRmin") {
            isR.min = std::stof(line);
        }
        else if (option == "isRmax") {
            isR.max = std::stof(line);
        }

        else if (option == "idRmin") {
            idR.min = std::stof(line);
        }
        else if (option == "idRmax") {
            idR.max = std::stof(line);
        }

        else if (option == "material") {
            auto selected = line;
            for(int i = 0; i < materialGenerator::typeCount; ++i)
            {
                if(selected == materialGenerator::typeString[i]) 
                {
                    matType = static_cast<materialGenerator::type>(i);
                    break;
                }
            }
        }
    }

    file.close();
}

/* helper functions */

static void eraseWhite(std::string& str)
{
    int i = 0;
    while (i < str.size() && (str[i] == ' ' || str[i] == '\t')) {
        ++i;
    }
    str.erase(0, i);
}

static std::string nextstrToSpace(std::string& str)
{
    eraseWhite(str);
    int i = 0;
    while (i < str.size() && str[i] != ' ')
        ++i;
    if (i == str.size())
        return str.substr();

    std::string res = str.substr(0, i);
    str = str.substr(i + 1, str.size() - i - 1);
    eraseWhite(str);
    return res;
}
