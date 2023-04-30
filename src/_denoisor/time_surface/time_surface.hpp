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

    class TimeSurface : public EventDenoisor {
    public:
        float_t decay;
        int16_t squareR;
        float_t threshold;

        std::vector<kore::Event> mPos;
        std::vector<kore::Event> mNeg;

        void regenerateParam() {
            mPos.resize(sizeX * sizeY);
            mNeg.resize(sizeX * sizeY);
        }

        bool calculateDensity(const int16_t &evtX, const int16_t &evtY, const int64_t &evtTimestamp, const bool &evtPolarity) {

            int16_t support       = 0;
            double_t difTimestamp = 0;

            auto &mCell = (evtPolarity == 1) ? mPos : mNeg;
            auto &prev  = mCell[evtX * sizeY + evtY];
            if (prev.timestamp() != 0) {
                for (size_t x = evtX - squareR; x <= evtX + squareR; x++) {
                    for (size_t y = evtY - squareR; y <= evtY + squareR; y++) {
                        if (x < 0 || x >= sizeX || y < 0 || y >= sizeY) {
                            continue;
                        }
                        auto &neighbor = mCell[x * sizeY + y];
                        if (neighbor.polarity() == evtPolarity) {
                            difTimestamp += 1 - exp(-(double)(evtTimestamp - neighbor.timestamp()) / decay);
                            support++;
                        }
                    }
                }
            }

            prev.timestamp_ = evtTimestamp;
            prev.x_         = evtX;
            prev.y_         = evtY;
            prev.polarity_  = evtPolarity;

            if (support != 0 && (1 - difTimestamp / (double_t)support) >= threshold) {
                return true;
            }

            return false;
        };
    };

}
