#include "denoisor.hpp"
#include "kore.hpp"

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <boost/circular_buffer.hpp>
#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/torch.h>

namespace edn {

    class EventDenoiseConvolutionNetwork : public EventDenoisor {
    public:
        std::string modelPath;
        int32_t _BATCHSIZE_;
        int32_t _DEPTH_;
        int16_t squareR;
        float_t threshold;
    
        size_t index = 0;
        int32_t _SQUARE_L_;
        std::vector<boost::circular_buffer<kore::Event>> mPos;
        std::vector<boost::circular_buffer<kore::Event>> mNeg;

        int32_t _ARRAY_;
        torch::jit::script::Module module;
        torch::Tensor batch;
        c10::Device device = torch::Device(torch::kCUDA, 0);

        void regenerateParam() {
            module     = torch::jit::load(modelPath, device);
            _SQUARE_L_ = 2 * squareR + 1;
            _ARRAY_    = _SQUARE_L_ * _SQUARE_L_ * 2 * _DEPTH_;
            batch      = torch::empty({_BATCHSIZE_, 2 * _DEPTH_, _SQUARE_L_, _SQUARE_L_});
            mPos.resize(sizeX * sizeY, boost::circular_buffer<kore::Event>(_DEPTH_));
            mNeg.resize(sizeX * sizeY, boost::circular_buffer<kore::Event>(_DEPTH_));
        }

        std::vector<float> buildInput(const int16_t &evtX, const int16_t &evtY, const int64_t &evtTimestamp, const bool &evtPolarity) {
            std::vector<float> patch(_ARRAY_, 0);

            int k = 0, layerSize = _ARRAY_ / (2 * _DEPTH_);
            // as for first event at each pixel, directly save its timestamp
            for (int i = evtX - squareR; i <= evtX + squareR; i++) {
                for (int j = evtY - squareR; j <= evtY + squareR; j++) {
                    int xn = i < 0 ? -i : (i >= sizeX ? 2 * sizeX - i - 1 : i);
                    int yn = j < 0 ? -j : (j >= sizeY ? 2 * sizeY - j - 1 : j);

                    int layer = 0;
                    int nnIdx = xn * sizeY + yn;
                    if (evtPolarity == 1) {
                        for (const auto &pos : mPos[nnIdx]) {
                            patch[(layer++) * layerSize + k] = buildTs(evtTimestamp, pos.timestamp());
                        }
                        for (const auto &neg : mNeg[nnIdx]) {
                            patch[(layer++) * layerSize + k] = buildTs(evtTimestamp, neg.timestamp());
                        }
                    } else {
                        for (const auto &neg : mNeg[nnIdx]) {
                            patch[(layer++) * layerSize + k] = buildTs(evtTimestamp, neg.timestamp());
                        }
                        for (const auto &pos : mPos[nnIdx]) {
                            patch[(layer++) * layerSize + k] = buildTs(evtTimestamp, pos.timestamp());
                        }
                    }
                    k++;
                }
            }

            if (evtPolarity == 1) {
                mPos[evtX * sizeY + evtY].push_front(kore::Event(evtTimestamp, evtY, evtX, evtPolarity));
            } else {
                mNeg[evtX * sizeY + evtY].push_front(kore::Event(evtTimestamp, evtY, evtX, evtPolarity));
            }

            return patch;
        };

        float buildTs(uint64_t evTs, uint64_t nnTs) {
            float_t maxTime = 5000000., minTime = 150.;
            float_t dT = evTs - nnTs;
            if (nnTs == 0 || dT >= maxTime) {
                dT = maxTime;
            }

            float_t ts = log(dT + 1) - log(minTime + 1);
            if (ts <= 0) {
                return 0;
            }
            
            return ts;
        };
    };
}
