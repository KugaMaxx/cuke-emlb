#include "denoisor.hpp"
#include "kore.hpp"

#include <immintrin.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <boost/circular_buffer.hpp>
#include <dv-sdk/module.hpp>

namespace edn {

    class DoubleWindowFilter : public EventDenoisor {
    public:
        int16_t squareR;
        int16_t threshold;

        uint8_t wLen;
        boost::circular_buffer<kore::Event> lastREvents; // real
        boost::circular_buffer<kore::Event> lastNEvents; // noise

        void regenerateParam() {
            lastREvents.resize(wLen);
            lastNEvents.resize(wLen);
        }

        bool calculateDensity(const int16_t &evtX, const int16_t &evtY, const int64_t &evtTimestamp, const bool &evtPolarity) {
            uint16_t count = 0;

            for (auto &lastR : lastREvents) {
                if (abs(evtX - lastR.x()) + abs(evtY - lastR.y()) <= squareR) {
                    count++;
                }
            }

            for (auto &lastN : lastNEvents) {
                if (abs(evtX - lastN.x()) + abs(evtY - lastN.y()) <= squareR) {
                    count++;
                }
            }

            if (count >= threshold) {
                lastREvents.push_back(kore::Event(evtTimestamp, evtY, evtX, evtPolarity));
                return true;
            }
            lastNEvents.push_back(kore::Event(evtTimestamp, evtY, evtX, evtPolarity));
            
            return false;
        };
    };

}
