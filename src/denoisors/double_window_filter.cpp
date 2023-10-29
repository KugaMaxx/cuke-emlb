#include "../module_utils.hpp"
#include "denoisors/double_window_filter.hpp"

namespace dv::module {

class DoubleWindowFilter : public dv::ModuleBase, public dv::noise::DoubleWindowFilter<> {
public:
    static const char *initDescription() {
        return "Double Window Filter.";
    };

    static void initInputs(dv::InputDefinitionList &in) {
        in.addEventInput("events");
    };

    static void initOutputs(dv::OutputDefinitionList &out) {
        out.addEventOutput("events");
    };

    static void initConfigOptions(dv::RuntimeConfig &config) {
        config.add("searchRadius", dv::ConfigOption::intOption("Max L1 distance.", 9, 1, 100));
        config.add("intThreshold", dv::ConfigOption::intOption("Threshold value (max of window length).", 1, 1, 100));
        config.add("bufferSize", dv::ConfigOption::intOption("Window length.", 36, 4, 100));

        config.setPriorityOptions({"searchRadius", "intThreshold", "bufferSize"});
    };

    DoubleWindowFilter() : 
        dv::ModuleBase(),
        dv::noise::DoubleWindowFilter<>(cv::Size{
            inputs.getEventInput("events").sizeX(),
            inputs.getEventInput("events").sizeY()}) {
        outputs.getEventOutput("events").setup(inputs.getEventInput("events"));
    };

    ~DoubleWindowFilter() {}

    void run() override {
        auto inEvent  = inputs.getEventInput("events").events();
        auto outEvent = outputs.getEventOutput("events").events();

        if (!inEvent) {
            return;
        }

        for (auto &event : inEvent) {
            bool isSignal = retain(event);

            if (isSignal) {
                outEvent << event;
            }
        }
        outEvent << dv::commit;
    };

    void configUpdate() override {
        mSearchRadius = config.getInt("searchRadius");
        mIntThreshold = config.getInt("intThreshold");
        mBufferSize = config.getInt("bufferSize");
        initialize();
    };
};

} // namespace dv::module

registerModuleClass(dv::module::DoubleWindowFilter)
