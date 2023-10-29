#include "../module_utils.hpp"
#include "denoisors/time_surface.hpp"

namespace dv::module {

class TimeSurface : public dv::ModuleBase, public dv::noise::TimeSurface<> {
public:
    static const char *initDescription() {
        return "Noise filter using the Time Surface.";
    };

    static void initInputs(dv::InputDefinitionList &in) {
        in.addEventInput("events");
    };

    static void initOutputs(dv::OutputDefinitionList &out) {
        out.addEventOutput("events");
    };

    static void initConfigOptions(dv::RuntimeConfig &config) {
        config.add("decay", dv::ConfigOption::intOption("Time decay.", 30000, 1, 1000000));
        config.add("searchRadius", dv::ConfigOption::intOption("Radius R of matrix for time surface.", 1, 1, 7));
        config.add("floatThreshold", dv::ConfigOption::floatOption("Threshold value for mean time surface value.", 0.3, 0, 1));

        config.setPriorityOptions({"decay", "squareR", "floatThreshold"});
    };

    TimeSurface() : 
        dv::ModuleBase(),
        dv::noise::TimeSurface<>(cv::Size{
            inputs.getEventInput("events").sizeX(),
            inputs.getEventInput("events").sizeY()}) {
        outputs.getEventOutput("events").setup(inputs.getEventInput("events"));
    };

    ~TimeSurface() {}

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
        mDecay = config.getInt("decay");
        mSearchRadius = config.getInt("searchRadius");
        mFloatThreshold = config.getFloat("floatThreshold");
        initialize();
    };
};

} // namespace dv::module

registerModuleClass(dv::module::TimeSurface)
