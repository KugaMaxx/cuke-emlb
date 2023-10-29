#include "../module_utils.hpp"
#include "denoisors/yang_noise.hpp"

namespace dv::module {

class YangNoise : public dv::ModuleBase, public dv::noise::YangNoise<> {
public:
    static const char *initDescription() {
        return "Noise filter using the YNoise.";
    };

    static void initInputs(dv::InputDefinitionList &in) {
        in.addEventInput("events");
    };

    static void initOutputs(dv::OutputDefinitionList &out) {
        out.addEventOutput("events");
    };

    static void initConfigOptions(dv::RuntimeConfig &config) {
        config.add("duration", dv::ConfigOption::intOption("Time delta for filter.", 10000, 1, 1000000));
        config.add("searchRadius", dv::ConfigOption::intOption("Radius R of matrix for density Matrix.", 1, 1, 7));
        config.add("intThreshold", dv::ConfigOption::intOption("Threshold value for density pixels.", 2, 0, 1000));

        config.setPriorityOptions({"duration", "searchRadius", "intThreshold"});
    };

    YangNoise() : 
        dv::ModuleBase(),
        dv::noise::YangNoise<>(cv::Size{
            inputs.getEventInput("events").sizeX(),
            inputs.getEventInput("events").sizeY()}) {
        outputs.getEventOutput("events").setup(inputs.getEventInput("events"));
    };

    ~YangNoise() {}

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
        mIntThreshold = config.getInt("intThreshold");
        initialize();
    };
};

} // namespace dv::module

registerModuleClass(dv::module::YangNoise)
