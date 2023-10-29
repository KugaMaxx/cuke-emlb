#pragma once

#include "../denoisor.hpp"

namespace dv::noise {

template<class EventStoreClass = dv::EventStore>
class YangNoise : public EventFilterBase<EventStoreClass> {
private:
    using EventType = typename EventStoreClass::value_type;

protected:
	int16_t mWidth;
	int16_t mHeight;
	int64_t mDuration;
	size_t mSearchRadius;
	size_t mIntThreshold;

	common::Offsets mOffsets;
	common::Matrix2d<int64_t> mTimestampMat;
	common::Matrix2d<uint8_t> mPolarityMat;

public:
	explicit YangNoise(
		const cv::Size &resolution, const dv::Duration duration = dv::Duration(2000),
		const size_t searchRadius = 1, const size_t intThreshold = 1) :
		mWidth(resolution.width),
		mHeight(resolution.height),
		mDuration(duration.count()),
		mSearchRadius(searchRadius),
		mIntThreshold(intThreshold),
		mTimestampMat(mWidth, mHeight),
		mPolarityMat(mWidth, mHeight) {
		this->initialize();
	}

	void initialize() {
		mOffsets.resize(mSearchRadius, mSearchRadius);
	}

	size_t calculateDensity(const EventType &event) {
		// Initialize density
		size_t density = 0;
	
		// Do spatio-temporal search
		for (const auto &delta : mOffsets) {
			int16_t x = event.x() + delta.x();
			int16_t y = event.y() + delta.y();

			if (x < 0 || y < 0) {
				continue;
			}

			if (x > mWidth - 1 || y > mHeight - 1) {
				continue;
			}

			if (event.timestamp() - mTimestampMat(x, y) > mDuration) {
				continue;
			}

			if (event.polarity() != mPolarityMat(x, y)) {
				continue;
			}

			density++;
		}

		return density;
	}

	bool evaluate(const EventType &event) {
		// calculate density in spatio-temporal neighborhood
		size_t density = calculateDensity(event);

		// evaluate
		bool isSignal = (density >= mIntThreshold);

		// update matrix
		mTimestampMat(event.x(), event.y()) = event.timestamp();
		mPolarityMat(event.x(), event.y())  = event.polarity();

		return isSignal;
	}

	inline bool retain(const EventType &event) noexcept override {
		return evaluate(event);
	}

	inline YangNoise &operator<<(const EventStoreClass &events) {
		accept(events);
		return *this;
	}
};

} // namespace dv::noise

