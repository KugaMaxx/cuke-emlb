#include "denoisor.hpp"
#include "kore.hpp"

#include <deque>
#include <immintrin.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <Eigen/Dense>
#include <dv-sdk/module.hpp>

namespace edn {

    class EventFlow : public EventDenoisor {
    public:
        int16_t squareR;
        int32_t deltaT;
        float_t threshold;

        std::deque<kore::Event> mDeque;

        bool calculateDensity(const int16_t &evtX, const int16_t &evtY, const int64_t &evtTimestamp, const bool &evtPolarity) {
            if (mDeque.size() != 0) {
                while (!mDeque.empty() && mDeque.back().timestamp() - mDeque.front().timestamp() >= deltaT) {
                    mDeque.pop_front();
                }
            }
            // Update
            mDeque.push_back(kore::Event(evtTimestamp, evtY, evtX, evtPolarity));

            std::vector<kore::Event> tmpCell;
            for (auto &mD : mDeque) {
                if (abs(evtX - mD.x()) <= squareR && abs(evtY - mD.y()) <= squareR) {
                    tmpCell.push_back(mD);
                }
            }

            size_t len = tmpCell.size();
            if (len < 3) {
                return false;
            }

            Eigen::MatrixXd A(len, 3);
            Eigen::VectorXd b(len, 1);
            for (size_t i = 0; i < tmpCell.size(); i++) {
                A(i, 0) = (double)tmpCell[i].x();
                A(i, 1) = (double)tmpCell[i].y();
                A(i, 2) = 1.0;
                b(i)    = ((double)tmpCell[i].timestamp() - evtTimestamp) * 0.001;
            }
            Eigen::Vector3d X = A.colPivHouseholderQr().solve(b);

            if (pow((pow(-1 / X[0], 2) + pow(-1 / X[1], 2)), 0.5) <= pow(10, threshold)) {
                return true;
            }

            return false;
        };
    };

}
