#include "denoisor.hpp"
#include "kore.hpp"

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <dv-sdk/module.hpp>

namespace edn {

    class YangNoise : public EventDenoisor {
    public:
        int32_t deltaT;
        int16_t squareR;
        int16_t threshold;

        size_t _NEIGH_;

        std::vector<kore::Event> mT;
        std::vector<std::pair<int16_t, int16_t>> offsets;

        void regenerateParam() {
            // resize memory cell
            mT.resize(sizeX * sizeY);
            // generate quick search array
            int32_t squareL = 2 * squareR + 1;
            _NEIGH_         = squareL * squareL;
            offsets.resize(_NEIGH_);
            for (size_t i = 0; i < _NEIGH_; i++) {
                offsets[i].first  = -squareR + i / squareL;
                offsets[i].second = -squareR + i % squareL;
            }
        }

        bool calculateDensity(const int16_t &evtX, const int16_t &evtY, const int64_t &evtTimestamp, const bool &evtPolarity) {

            int16_t sum           = 0;
            int16_t lInfNorm      = 0;
            int64_t timeToCompare = evtTimestamp - deltaT;

            for (uint32_t i = 0; i < _NEIGH_; i++) {
                uint32_t ny = evtY + offsets[i].second;
                uint32_t nx = evtX + offsets[i].first;
                if (nx >= 0 && ny >= 0 && nx < sizeX && ny < sizeY) {
                    auto &neighbor = mT[nx * sizeY + ny];
                    if (evtPolarity == neighbor.polarity() && neighbor.timestamp() > timeToCompare) {
                        if (offsets[i].second == -squareR) {
                            lInfNorm = std::max(lInfNorm, sum);
                            sum      = 0;
                        }
                        sum++;
                    }
                }
            }

            // Update
            auto &mCell = mT[evtX * sizeY + evtY];
            mCell.timestamp_ = evtTimestamp;
            mCell.x_      = evtX;
            mCell.y_      = evtY;
            mCell.polarity_ = evtPolarity;

            if (lInfNorm >= threshold) {
                return true;
            }
            
            return false;
        };
    };

}
