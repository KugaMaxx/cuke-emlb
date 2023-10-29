#include "../module_utils.hpp"
#include "denoisors/reclusive_event_denoisor.hpp"

namespace dv::module {

class ReclusiveEventDenoisor : public dv::ModuleBase, public dv::noise::ReclusiveEventDenoisor<> {
public:
    static const char *initDescription() {
        return "Reclusive filter using state space and deriche blur.";
    };

    static void initInputs(dv::InputDefinitionList &in) {
        in.addEventInput("events");
    };

    static void initOutputs(dv::OutputDefinitionList &out) {
        out.addEventOutput("events");
    };

    static void initConfigOptions(dv::RuntimeConfig &config) {
        config.add("sigmaS", dv::ConfigOption::floatOption("Spatial blur coefficient.", 0.7, 0.1, 3.0));
        config.add("sigmaT", dv::ConfigOption::intOption("Time sigma.", 1, 1, 5));
        config.add("samplarT", dv::ConfigOption::floatOption("Log scale to slice event stream.", -0.8, -2.0, 2.0));
        config.add("floatThreshold", dv::ConfigOption::floatOption("Threshold value.", 0.2, -1.0, 3.0));

        config.setPriorityOptions({"sigmaT", "sigmaS", "samplarT", "floatThreshold"});
    };

    ReclusiveEventDenoisor() : 
        dv::ModuleBase(),
        dv::noise::ReclusiveEventDenoisor<>(cv::Size{
            inputs.getEventInput("events").sizeX(),
            inputs.getEventInput("events").sizeY()}) {
        outputs.getEventOutput("events").setup(inputs.getEventInput("events"));
    };

    ~ReclusiveEventDenoisor() {}

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
        sigmaS    = config.getFloat("sigmaS");
        sigmaT    = config.getInt("sigmaT");
        samplarT  = config.getFloat("samplarT");
        floatThreshold = config.getFloat("floatThreshold");
        initialize();
    };
};

} // namespace dv::module

registerModuleClass(dv::module::ReclusiveEventDenoisor)
