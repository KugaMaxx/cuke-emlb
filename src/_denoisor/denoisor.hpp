#ifndef EDN_H
#define EDN_H

#include <stdlib.h>
#include <iostream>

namespace edn {
    
class EventDenoisor {
public:
    int16_t sizeX;
    int16_t sizeY;
    size_t  _LENGTH_;  // sizeX * sizeY
};

}

#endif
