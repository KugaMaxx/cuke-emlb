#pragma once

#include "../denoisor.hpp"

#include <boost/circular_buffer.hpp>

namespace dv::noise {

template<class EventStoreClass = dv::EventStore>
class DoubleWindowFilter : public EventFilterBase<EventStoreClass> {
private:
    using EventType = typename EventStoreClass::value_type;
	using CircularBuffer = boost::circular_buffer<EventType>;

protected:
	int16_t mWidth;
	int16_t mHeight;
	size_t mSearchRadius;
	size_t mIntThreshold;

    size_t mBufferSize;
	CircularBuffer lastRealEvents;
	CircularBuffer lastNoiseEvents;

public:
    explicit DoubleWindowFilter(
        const cv::Size &resolution, const size_t bufferSize = 36,
        const size_t searchRadius = 9, const size_t intThreshold = 1) :
		mWidth(resolution.width),
		mHeight(resolution.height),
        mBufferSize(bufferSize),
        mSearchRadius(searchRadius),
		mIntThreshold(intThreshold) {
        this->initialize();
	}

    void initialize() {
        lastRealEvents.resize(mBufferSize);
        lastNoiseEvents.resize(mBufferSize);
    }

    size_t countNearbyEvents(const EventType &event) {
        size_t count = 0;

        // count events in real events window
        for (const auto &real : lastRealEvents) {
            if (abs(event.x() - real.x()) + abs(event.y() - real.y()) <= mSearchRadius) {
                count = count + 1;
                if (count >= mIntThreshold) {
                    return count;
                }
            }
        }

        // count events in noise events window
        for (const auto &noise : lastNoiseEvents) {
            if (abs(event.x() - noise.x()) + abs(event.y() - noise.y()) <= mSearchRadius) {
                count = count + 1;
                if (count >= mIntThreshold) {
                    return count;
                }
            }
        }

        return count;
    }

	bool evaluate(const EventType &event) {
		// count related events in window
        size_t count = countNearbyEvents(event);

        // evaluate
		bool isSignal = (count >= mIntThreshold);

        // update
        if (isSignal) {
            lastRealEvents.push_back(event);
        } else {
            lastNoiseEvents.push_back(event);
        }
        
		return isSignal;
	}

	inline bool retain(const EventType &event) noexcept override {
		return evaluate(event);
	}

	inline DoubleWindowFilter &operator<<(const EventStoreClass &events) {
		accept(events);
		return *this;
	}
};

}