#pragma once

#include "../denoisor.hpp"

#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/torch.h>

namespace fs = std::filesystem;

namespace dv::noise {

template<class EventStoreClass = dv::EventStore>
class MultiLayerPerceptronFilter : public EventFilterBase<EventStoreClass> {
private:
    using EventType = typename EventStoreClass::value_type;

protected:
	int16_t mWidth;
	int16_t mHeight;
    fs::path mModelPath;
	bool mModelIsLoad;
    int32_t mBatchSize;
	int64_t mDuration;
    double mFloatThreshold;

    const int16_t mInputDepth = 2;
    const int16_t mInputWidth = 7;
    const int16_t mInputHeight = 7;
	const int16_t mInputArea = mInputWidth * mInputHeight;
    const int16_t mInputVolume = mInputDepth * mInputWidth * mInputHeight;

	common::Offsets mOffsets;
	common::Matrix2d<EventType> mTimeSurface;
	dv::StreamSlicer<EventStoreClass> mSlicer;

	torch::Device mDevice;
	torch::jit::script::Module mPreTrainedModel;

public:
    explicit MultiLayerPerceptronFilter(
        const cv::Size &resolution, const fs::path modelPath = fs::path(), const size_t batchSize = 5000, 
        const dv::Duration duration = dv::Duration(100000), const double floatThreshold = 0.8, const size_t deviceId = 0) :
		mWidth(resolution.width),
		mHeight(resolution.height),
		mModelPath(modelPath),
		mBatchSize(batchSize),
        mDuration(duration.count()),
		mOffsets(mInputWidth / 2, mInputHeight / 2),
		mFloatThreshold(floatThreshold),
        mTimeSurface(mWidth, mHeight),
		mDevice(torch::kCUDA, deviceId) {
		// this->initialize();
	}

	void initialize() {
		if (!mModelPath.empty() && !mModelIsLoad) {
			mPreTrainedModel = torch::jit::load(mModelPath, mDevice);
			mModelIsLoad = true;
		}
	}

	double logarithmicTimeDiff(const int64_t &fromTime, const int64_t &toTime) {
		double deltaTime = fromTime - toTime;
		double maxDeltaTime = 5000000.;
		double minDeltaTime = 150.;
		
		deltaTime = std::min(deltaTime, maxDeltaTime);
		deltaTime = std::max(deltaTime, minDeltaTime);

		return std::log((deltaTime + 1) / (minDeltaTime + 1));
	}

	torch::Tensor buildInputTensor(const EventStoreClass &store) {
        // create empty input tensor
		torch::Tensor inputTensor = torch::empty({mBatchSize, mInputVolume});

        // build input tensor
		size_t batchInd = 0;
		for (const auto &event : store) {
			// construct single input
			size_t k = 0;
			std::vector<double> single(mInputVolume, 0);

            // look up neighborhood
			for (const auto &delta : mOffsets) {
				int16_t x = event.x() + delta.x();
				int16_t y = event.y() + delta.y();

                if (x < 0 || y < 0 || x > mWidth - 1 || y > mHeight - 1) {
                    single[k] = 0L;
                    single[k + mInputArea] = 0L;
                } else {
                    single[k] = 1L - (event.timestamp() - mTimeSurface(x, y).timestamp()) / mDuration;
                    single[k + mInputArea] = 2 * event.polarity() - 1;
                }

                k = k + 1;
			}
            inputTensor[batchInd++] = torch::from_blob(single.data(), {mInputVolume}, torch::kFloat);

			// update
            mTimeSurface(event.x(), event.y()) = event;
		}

		return inputTensor;
	}

	[[nodiscard]] EventStoreClass generateEvents() {
		if (!this->buffer.isEmpty() && mModelIsLoad) {
			this->numIncomingEvents += this->buffer.size();

			std::shared_ptr<typename EventStoreClass::packet_type> packet
				= std::make_shared<typename EventStoreClass::packet_type>();
			packet->elements.reserve(this->buffer.size());

			// clear former slicer job
			mSlicer.removeJob(0);
            
			// do every batch size
			mSlicer.doEveryNumberOfElements(mBatchSize, [this, packet](EventStoreClass &store) {
				torch::Tensor inputTensor = buildInputTensor(store);
				torch::Tensor outputTensor = mPreTrainedModel.forward({inputTensor.to(mDevice)}).toTensor().to(torch::kCPU);
				for (size_t i = 0; i < outputTensor.size(0); i++) {
					if (outputTensor[i][0].item<double>() >= mFloatThreshold) {
						packet->elements.push_back(store.at(i));
					}
				}
			});
			mSlicer.accept(this->buffer);

			packet->elements.shrink_to_fit();
			this->numOutgoingEvents    += packet->elements.size();
			this->highestProcessedTime = this->buffer.getHighestTime();

			this->buffer = EventStoreClass{};
			return EventStoreClass(std::const_pointer_cast<typename EventStoreClass::const_packet_type>(packet));
		}

		return {};
	}

	inline bool retain(const EventType &event) noexcept override {
		return true;
	}

	inline MultiLayerPerceptronFilter &operator<<(const EventStoreClass &events) {
		accept(events);
		return *this;
	}
};

} // namespace dv::noise

