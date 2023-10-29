#pragma once
#define FMT_HEADER_ONLY

#include <utility>

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <fmt/core.h>
#include <dv-toolkit/toolkit.hpp>
#include <dv-processing/processing.hpp>

namespace py  = pybind11;

namespace pybind11::detail {

template<>
struct type_caster<cv::Size> {
	PYBIND11_TYPE_CASTER(cv::Size, _("tuple_xy"));

	bool load(handle obj, bool) {
		if (!py::isinstance<py::tuple>(obj)) {
			std::logic_error("Size(width,height) should be a tuple!");
			return false;
		}

		auto pt = reinterpret_borrow<py::tuple>(obj);
		if (pt.size() != 2) {
			std::logic_error("Size(width,height) tuple should be size of 2");
			return false;
		}

		value = cv::Size(pt[0].cast<int>(), pt[1].cast<int>());
		return true;
	}

	static handle cast(const cv::Size &resolution, return_value_policy, handle) {
		return py::make_tuple(resolution.width, resolution.height).release();
	}
};

} // namespace pybind11::detail
