#pragma once

#include "../denoisor.hpp"

#include <deque>
#include <numeric>
#include <Eigen/Dense>

namespace dv::noise {

template<class EventStoreClass = dv::EventStore>
class EventFlow : public EventFilterBase<EventStoreClass> {
private:
    using EventType = typename EventStoreClass::value_type;

protected:
	int16_t mWidth;
	int16_t mHeight;
	int64_t mDuration;
    size_t mSearchRadius;
    float_t mFloatThreshold;

    std::deque<EventType> mDeque;

public:
	explicit EventFlow(
		const cv::Size &resolution, const dv::Duration duration = dv::Duration(2000),
		const size_t searchRadius = 1, const double floatThreshold = 20.0) :
		mWidth(resolution.width),
		mHeight(resolution.height),
        mSearchRadius(searchRadius),
        mFloatThreshold(floatThreshold) {
        this->initialize();
	}

    void initialize() {
    }

	size_t fitEventFlow(const EventType &event) {
        // default flow value will be infinity
        double flow = std::numeric_limits<double>::max();
        
        // search spatio-temporal related events
        std::vector<EventType> candidateEvents;
        for (const auto &deque : mDeque) {
            if ((abs(event.x() - deque.x()) <= mSearchRadius) && (abs(event.y() - deque.y()) <= mSearchRadius)) {
                candidateEvents.push_back(deque);
            }
        }

        // calculate flow
        if (candidateEvents.size() > 3) {
            Eigen::MatrixXd A(candidateEvents.size(), 3);
            Eigen::MatrixXd b(candidateEvents.size(), 1);
            for (size_t i = 0; i < candidateEvents.size(); i++) {
                A(i, 0) = candidateEvents[i].x();
                A(i, 1) = candidateEvents[i].y();
                A(i, 2) = 1.0;
                b(i)    = (candidateEvents[i].timestamp() - event.timestamp()) * 1E-3;
            }

            // solve
            Eigen::Vector3d X = A.colPivHouseholderQr().solve(b);
            flow = pow((pow(-1 / X[0], 2) + pow(-1 / X[1], 2)), 0.5);
        }

        return flow;
	}

	bool evaluate(const EventType &event) {
		// calculate density in spatio-temporal neighborhood
		size_t flow = fitEventFlow(event);

		// evaluate
		bool isSignal = (flow <= mFloatThreshold);

		// update deque
        while (!mDeque.empty()) {
            if (event.timestamp() - mDeque.front().timestamp() >= mDuration) {
                mDeque.pop_front();
            } else {
                break;
            }
        }
        mDeque.push_back(event);

		return isSignal;
	}

	inline bool retain(const EventType &event) noexcept override {
		return evaluate(event);
	}

	inline EventFlow &operator<<(const EventStoreClass &events) {
		accept(events);
		return *this;
	}
};

} // namespace dv::noise

