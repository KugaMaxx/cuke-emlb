#include "denoisor.hpp"
#include "kore.hpp"

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/torch.h>

namespace edn {

    class MultiLayerPerceptronFilter : public EventDenoisor {
    public:
        std::string modelPath;
        int32_t _BATCHSIZE_;
        float_t decay;
        int16_t squareR;
        float_t threshold;

        size_t index = 0;
        std::vector<kore::Event> mT;

        int32_t _ARRAY_;
        torch::jit::script::Module module;
        torch::Tensor batch;
        c10::Device device = torch::Device(torch::kCUDA, 0);

        void regenerateParam() {
            module  = torch::jit::load(modelPath, device);
            _ARRAY_ = (2 * squareR + 1) * (2 * squareR + 1) * 2;
            batch = torch::empty({_BATCHSIZE_, _ARRAY_});
            mT.resize(sizeX * sizeY);
        }

        std::vector<float> buildInput(const int16_t &evtX, const int16_t &evtY, const int64_t &evtTimestamp, const bool &evtPolarity) {
            int k = 0;
            std::vector<float> patch(_ARRAY_, 0);

            for (int x = evtX - squareR; x <= evtX + squareR; x++) {
                for (int y = evtY - squareR; y <= evtY + squareR; y++) {
                    auto &mNeigh = mT[x * sizeY + y];
                    if (x < 0 || x >= sizeX || y < 0 || y >= sizeY || mNeigh.timestamp() <= 0 || evtTimestamp - mNeigh.timestamp() > decay) {
                        patch[k]            = 0L;
                        patch[k + _LENGTH_] = 0L;
                    } else {
                        patch[k]            = 1L - ((evtTimestamp - mNeigh.timestamp()) / decay);
                        patch[k + _LENGTH_] = evtPolarity;
                    }
                    k++;
                }
            }

            // Update
            auto &mCell      = mT[evtX * sizeY + evtY];
            mCell.timestamp_ = evtTimestamp;
            mCell.x_         = evtX;
            mCell.y_         = evtY;
            mCell.polarity_  = 2 * (int)evtPolarity - 1;

            return patch;
        };
    };

}
