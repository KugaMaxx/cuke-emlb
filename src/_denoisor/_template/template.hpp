#include "denoisor.hpp"
#include "kore.hpp"

#include <immintrin.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <dv-sdk/module.hpp>

namespace edn {

    class Template : public EventDenoisor {
    public:
        float_t params;

        bool calculateDensity(const int16_t &evtX, const int16_t &evtY, const int64_t &evtTimestamp, const bool &evtPolarity) {
            return false;
        };
    };

}
