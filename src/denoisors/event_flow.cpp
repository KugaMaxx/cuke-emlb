#include "../module_utils.hpp"
#include "denoisors/event_flow.hpp"

namespace dv::module {

class EventFlow : public dv::ModuleBase, public dv::noise::EventFlow<> {
public:
    static const char *initDescription() {
        return "Noise filter using the EventFlow.";
    };

    static void initInputs(dv::InputDefinitionList &in) {
        in.addEventInput("events");
    };

    static void initOutputs(dv::OutputDefinitionList &out) {
        out.addEventOutput("events");
    };

    static void initConfigOptions(dv::RuntimeConfig &config) {
        config.add("duration", dv::ConfigOption::intOption("Time delta for filter.", 3000, 1, 10000));
        config.add("searchRadius", dv::ConfigOption::intOption("Radius R of matrix for density Matrix.", 1, 1, 7));
        config.add("floatThreshold", dv::ConfigOption::floatOption("Threshold of velocity.", 2, 0, 10));

        config.setPriorityOptions({"duration", "searchRadius", "floatThreshold"});
    };

    EventFlow() : 
        dv::ModuleBase(),
        dv::noise::EventFlow<>(cv::Size{
            inputs.getEventInput("events").sizeX(),
            inputs.getEventInput("events").sizeY()}) {
        outputs.getEventOutput("events").setup(inputs.getEventInput("events"));
    };

    ~EventFlow() {}

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
        mDuration = config.getInt("duration");
        mSearchRadius = config.getInt("searchRadius");
        mFloatThreshold = config.getInt("floatThreshold");
        initialize();
    };
};

} // namespace dv::module

registerModuleClass(dv::module::EventFlow)
