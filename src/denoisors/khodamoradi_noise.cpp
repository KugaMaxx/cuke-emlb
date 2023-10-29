#include "../module_utils.hpp"
#include "denoisors/khodamoradi_noise.hpp"

namespace dv::module {

class KhodamoradiNoise : public dv::ModuleBase, public dv::noise::KhodamoradiNoise<> {
public:
    static const char *initDescription() {
        return "Noise filter using the KhodamoradiNoise.";
    };

    static void initInputs(dv::InputDefinitionList &in) {
        in.addEventInput("events");
    };

    static void initOutputs(dv::OutputDefinitionList &out) {
        out.addEventOutput("events");
    };

    static void initConfigOptions(dv::RuntimeConfig &config) {
        config.add("duration", dv::ConfigOption::intOption("Time delta for filter.", 1000, 1, 1000000));
        config.add("intThreshold", dv::ConfigOption::intOption("Number of supporting pixels.", 1, 0, 10));

        config.setPriorityOptions({"duration", "intThreshold"});
    };

    KhodamoradiNoise() : 
        dv::ModuleBase(),
        dv::noise::KhodamoradiNoise<>(cv::Size{
            inputs.getEventInput("events").sizeX(),
            inputs.getEventInput("events").sizeY()}) {
        outputs.getEventOutput("events").setup(inputs.getEventInput("events"));
    };

    ~KhodamoradiNoise() {}

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
        mIntThreshold = config.getInt("intThreshold");
        initialize();
    };
};

} // namespace dv::module

registerModuleClass(dv::module::KhodamoradiNoise)
