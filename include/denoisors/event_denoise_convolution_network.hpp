#pragma once

#include "../denoisor.hpp"

#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <boost/circular_buffer.hpp>

namespace dv::noise {

template<class EventStoreClass = dv::EventStore>
class EventDenoiseConvolutionNetwork : public EventFilterBase<EventStoreClass> {
private:
    using EventType = typename EventStoreClass::value_type;
	using CircularBuffer = boost::circular_buffer<EventType>;

protected:
	int16_t mWidth;
	int16_t mHeight;
    fs::path mModelPath;
    int32_t mBatchSize;
    double mFloatThreshold;
  
    const int16_t mInputDepth = 2;
    const int16_t mInputWidth = 25;
    const int16_t mInputHeight = 25;
	const int16_t mInputArea = mInputWidth * mInputHeight;
	const int16_t mInputVolume = 2 * mInputDepth * mInputWidth * mInputHeight;

	common::Offsets mOffsets;
	common::Matrix2d<CircularBuffer> mPosBuffer;
	common::Matrix2d<CircularBuffer> mNegBuffer;

	torch::Device mDevice;
	torch::jit::script::Module mPreTrainedModel;

public:
    explicit EventDenoiseConvolutionNetwork(
        const cv::Size &resolution, const fs::path modelPath, 
		const size_t batchSize = 1000, const double floatThreshold = 0.5, const size_t deviceId = 0) :
		mWidth(resolution.width),
		mHeight(resolution.height),
		mBatchSize(batchSize),
		mPosBuffer(mWidth, mHeight),
		mNegBuffer(mWidth, mHeight),
		mOffsets(mInputWidth / 2, mInputHeight / 2),
		mFloatThreshold(floatThreshold),
		mDevice(torch::kCUDA, deviceId),
		mPreTrainedModel(torch::jit::load(modelPath, mDevice)) {
		this->mPosBuffer.assign(mWidth * mHeight, CircularBuffer(mInputDepth));
		this->mNegBuffer.assign(mWidth * mHeight, CircularBuffer(mInputDepth));
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
		torch::Tensor inputTensor = torch::empty({mBatchSize, 2 * mInputDepth, mInputWidth, mInputHeight});
		
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
				x = x < 0 ? -x : (x >= mWidth ? 2 * mWidth - x - 1 : x);
				y = y < 0 ? -y : (y >= mHeight ? 2 * mHeight - y - 1 : y);
				
				// explore all buffer
				size_t i = 0;
				if (event.polarity() == 1) {
					for (const auto &buffer : mPosBuffer(x, y)) {
						single[(i++) * mInputArea + k] = logarithmicTimeDiff(event.timestamp(), buffer.timestamp());
					}
					for (const auto &buffer : mNegBuffer(x, y)) {
						single[(i++) * mInputArea + k] = logarithmicTimeDiff(event.timestamp(), buffer.timestamp());;
					}
				} else {
					for (const auto &buffer : mNegBuffer(x, y)) {
						single[(i++) * mInputArea + k] = logarithmicTimeDiff(event.timestamp(), buffer.timestamp());;
					}
					for (const auto &buffer : mPosBuffer(x, y)) {
						single[(i++) * mInputArea + k] = logarithmicTimeDiff(event.timestamp(), buffer.timestamp());;
					}
				}

				k = k + 1;
			}
			inputTensor[batchInd++] = torch::from_blob(single.data(), {2 * mInputDepth, mInputWidth, mInputHeight}, torch::kFloat);

			// update
			if (event.polarity() == 1) {
				mPosBuffer(event.x(), event.y()).push_front(event);
			} else {
				mNegBuffer(event.x(), event.y()).push_front(event);
			}
		}

		return inputTensor;
	}

	[[nodiscard]] EventStoreClass generateEvents() {
		if (!this->buffer.isEmpty()) {
			this->numIncomingEvents += this->buffer.size();

			std::shared_ptr<typename EventStoreClass::packet_type> packet
				= std::make_shared<typename EventStoreClass::packet_type>();
			packet->elements.reserve(this->buffer.size());

			// clear former slicer job
			dv::StreamSlicer<EventStoreClass> slicer;

			// do every batch size
			slicer.doEveryNumberOfElements(mBatchSize, [this, packet](EventStoreClass &store) {
				torch::Tensor inputTensor = buildInputTensor(store);
				torch::Tensor outputTensor = mPreTrainedModel.forward({inputTensor.to(mDevice)}).toTensor().to(torch::kCPU);
				for (size_t i = 0; i < outputTensor.size(0); i++) {
					if (outputTensor[i][0].item<double>() >= mFloatThreshold) {
						packet->elements.push_back(store.at(i));
					}
				}
			});

			// run inference
			slicer.accept(this->buffer);

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

	inline EventDenoiseConvolutionNetwork &operator<<(const EventStoreClass &events) {
		accept(events);
		return *this;
	}
};

} // namespace dv::noise

