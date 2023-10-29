#pragma once

#include "../denoisor.hpp"

namespace dv::noise {

template<class EventStoreClass = dv::EventStore>
class TimeSurface : public EventFilterBase<EventStoreClass> {
private:
    using EventType = typename EventStoreClass::value_type;

protected:
	int16_t mWidth;
	int16_t mHeight;
	size_t mSearchRadius;
	double mDecay;
	double mFloatThreshold;

	common::Offsets mOffsets;
	common::Matrix2d<EventType> mPos;
	common::Matrix2d<EventType> mNeg;

public:
	explicit TimeSurface(
		const cv::Size &resolution, const dv::Duration decay = dv::Duration(20000), 
		const size_t searchRadius = 1, const double floatThreshold = 0.2) :
		mWidth(resolution.width),
		mHeight(resolution.height),
		mDecay(decay.count()),
		mSearchRadius(searchRadius),
		mFloatThreshold(floatThreshold),
		mPos(mWidth, mHeight),
		mNeg(mWidth, mHeight) {
		this->initialize();
	}

	void initialize() {
		mOffsets.resize(mSearchRadius, mSearchRadius);
	}

	double fitTimeSurface(const EventType &event) {
		// initialize
		size_t support = 0;
		double diffTime = 0;

		// search in radius
		auto &cell = (event.polarity()) ? mPos : mNeg;
		for (const auto &delta : mOffsets) {
			int16_t x = event.x() + delta.x();
			int16_t y = event.y() + delta.y();

			if (x < 0 || y < 0) {
				continue;
			}

			if (x > mWidth - 1 || y > mHeight - 1) {
				continue;
			}

			if (cell(x, y).timestamp() == 0) {
				continue;
			}

			diffTime += exp((cell(x, y).timestamp() - event.timestamp()) / mDecay);

			support = support + 1;
		}

		// calculate surface
		double surface = (support == 0) ? 0 : diffTime / support;
		
		return surface;
	}

	bool evaluate(const EventType &event) {
		// calculate density in spatio-temporal neighborhood
		double surface = fitTimeSurface(event);

		// evaluate
		bool isSignal = (surface >= mFloatThreshold);

		// update matrix
		if (event.polarity() == 1) {
			mPos(event.x(), event.y()) = event;
		} else {
			mNeg(event.x(), event.y()) = event;
		}

		return isSignal;
	}

	inline bool retain(const EventType &event) noexcept override {
		return evaluate(event);
	}

	inline TimeSurface &operator<<(const EventStoreClass &events) {
		accept(events);
		return *this;
	}
};

} // namespace dv::noise

