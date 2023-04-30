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

    class KhodamoradiNoise : public EventDenoisor {
    public:
        int32_t deltaT;
        int16_t threshold;
        std::vector<kore::Event> xCols;
        std::vector<kore::Event> yRows;

        void regenerateParam() {
            xCols.resize(sizeX);
            yRows.resize(sizeY);
        }

        bool calculateDensity(const int16_t &evtX, const int16_t &evtY, const int64_t &evtTimestamp, const bool &evtPolarity) {

            int16_t support = 0;
            bool xMinusOne  = (evtX > 0);
            bool xPlusOne   = (evtX < (sizeX - 1));
            bool yMinusOne  = (evtY > 0);
            bool yPlusOne   = (evtY < (sizeY - 1));

            if (xMinusOne) {
                auto &xPrev = xCols[evtX - 1];
                if ((evtTimestamp - xPrev.timestamp()) <= deltaT && evtPolarity == xPrev.polarity()) {
                    if ((yMinusOne && (xPrev.y() == (evtY - 1))) || (xPrev.y() == evtY) || (yPlusOne && (xPrev.y() == (evtY + 1)))) {
                        support++;
                    }
                }
            }

            auto &xCell = xCols[evtX];
            if ((evtTimestamp - xCell.timestamp()) <= deltaT && evtPolarity == xCell.polarity()) {
                if ((yMinusOne && (xCell.y() == (evtY - 1))) || (yPlusOne && (xCell.y() == (evtY + 1)))) {
                    support++;
                }
            }

            if (xPlusOne) {
                auto &xNext = xCols[evtX + 1];
                if ((evtTimestamp - xNext.timestamp()) <= deltaT && evtPolarity == xNext.polarity()) {
                    if ((yMinusOne && (xNext.y() == (evtY - 1))) || (xNext.y() == evtY) || (yPlusOne && (xNext.y() == (evtY + 1)))) {
                        support++;
                    }
                }
            }

            if (yMinusOne) {
                auto &yPrev = yRows[evtY - 1];
                if ((evtTimestamp - yPrev.timestamp()) <= deltaT && evtPolarity == yPrev.polarity()) {
                    if ((xMinusOne && (yPrev.x() == (evtX - 1))) || (yPrev.x() == evtX) || (xPlusOne && (yPrev.x() == (evtX + 1)))) {
                        support++;
                    }
                }
            }

            auto &yCell = yRows[evtY];
            if ((evtTimestamp - yCell.timestamp()) <= deltaT && evtPolarity == yCell.polarity()) {
                if ((xMinusOne && (yCell.x() == (evtX - 1))) || (xPlusOne && (yCell.x() == (evtX + 1)))) {
                    support++;
                }
            }

            if (yPlusOne) {
                auto &yNext = yRows[evtY + 1];
                if ((evtTimestamp - yNext.timestamp()) <= deltaT && evtPolarity == yNext.polarity()) {
                    if ((xMinusOne && (yNext.x() == (evtX - 1))) || (yNext.x() == evtX) || (xPlusOne && (yNext.x() == (evtX + 1)))) {
                        support++;
                    }
                }
            }

            // Update
            xCell.timestamp_ = evtTimestamp;
            xCell.polarity_  = evtPolarity;
            xCell.y_         = evtY;

            yCell.timestamp_ = evtTimestamp;
            yCell.polarity_  = evtPolarity;
            yCell.x_         = evtX;

            if (support >= threshold) {
                return true;
            }

            return false;
        };
    };
}
