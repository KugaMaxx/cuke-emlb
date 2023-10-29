#pragma once

#include "../denoisor.hpp"

namespace dv::noise {

template<class EventStoreClass = dv::EventStore>
class KhodamoradiNoise : public EventFilterBase<EventStoreClass> {
private:
    using EventType = typename EventStoreClass::value_type;

protected:
	int16_t mWidth;
	int16_t mHeight;
	int64_t mDuration;
	size_t mIntThreshold;

	std::vector<EventType> xCols;
	std::vector<EventType> yRows;

public:
	explicit KhodamoradiNoise(
		const cv::Size &resolution, const dv::Duration duration = dv::Duration(2000), 
		const size_t intThreshold = 1) :
		mWidth(resolution.width),
		mHeight(resolution.height),
		mDuration(duration.count()),
		mIntThreshold(intThreshold),
		xCols(mWidth), 
		yRows(mHeight) {
		this->initialize();
	}

	void initialize() {
	}

	size_t searchCorrelation(const EventType &event) {
		size_t support = 0;
		bool xMinusOne  = (event.x() > 0);
		bool xPlusOne   = (event.x() < (mWidth - 1));
		bool yMinusOne  = (event.y() > 0);
		bool yPlusOne   = (event.y() < (mHeight - 1));

		if (xMinusOne) {
			auto &xPrev = xCols[event.x() - 1];
			if ((event.timestamp() - xPrev.timestamp()) <= mDuration && event.polarity() == xPrev.polarity()) {
				if ((yMinusOne && (xPrev.y() == (event.y() - 1))) || (xPrev.y() == event.y()) || (yPlusOne && (xPrev.y() == (event.y() + 1)))) {
					support++;
				}
			}
		}

		auto &xCell = xCols[event.x()];
		if ((event.timestamp() - xCell.timestamp()) <= mDuration && event.polarity() == xCell.polarity()) {
			if ((yMinusOne && (xCell.y() == (event.y() - 1))) || (yPlusOne && (xCell.y() == (event.y() + 1)))) {
				support++;
			}
		}

		if (xPlusOne) {
			auto &xNext = xCols[event.x() + 1];
			if ((event.timestamp() - xNext.timestamp()) <= mDuration && event.polarity() == xNext.polarity()) {
				if ((yMinusOne && (xNext.y() == (event.y() - 1))) || (xNext.y() == event.y()) || (yPlusOne && (xNext.y() == (event.y() + 1)))) {
					support++;
				}
			}
		}

		if (yMinusOne) {
			auto &yPrev = yRows[event.y() - 1];
			if ((event.timestamp() - yPrev.timestamp()) <= mDuration && event.polarity() == yPrev.polarity()) {
				if ((xMinusOne && (yPrev.x() == (event.x() - 1))) || (yPrev.x() == event.x()) || (xPlusOne && (yPrev.x() == (event.x() + 1)))) {
					support++;
				}
			}
		}

		auto &yCell = yRows[event.y()];
		if ((event.timestamp() - yCell.timestamp()) <= mDuration && event.polarity() == yCell.polarity()) {
			if ((xMinusOne && (yCell.x() == (event.x() - 1))) || (xPlusOne && (yCell.x() == (event.x() + 1)))) {
				support++;
			}
		}

		if (yPlusOne) {
			auto &yNext = yRows[event.y() + 1];
			if ((event.timestamp() - yNext.timestamp()) <= mDuration && event.polarity() == yNext.polarity()) {
				if ((xMinusOne && (yNext.x() == (event.x() - 1))) || (yNext.x() == event.x()) || (xPlusOne && (yNext.x() == (event.x() + 1)))) {
					support++;
				}
			}
		}
		
		return support;
	}

	bool evaluate(const EventType &event) {
		// calculate density in spatio-temporal neighborhood
		size_t support = searchCorrelation(event);

		// evaluate
		bool isSignal = (support >= mIntThreshold);

		// update matrix
		xCols[event.x()] = event;
		yRows[event.y()] = event;

		return isSignal;
	}

	inline bool retain(const EventType &event) noexcept override {
		return evaluate(event);
	}

	inline KhodamoradiNoise &operator<<(const EventStoreClass &events) {
		accept(events);
		return *this;
	}
};

} // namespace dv::noise

