#include "khodamoradi_noise.hpp"

namespace kdv {

    class KhodamoradiNoise : public edn::KhodamoradiNoise, public dv::ModuleBase {
    public:
        static const char *initDescription() {
            return "Noise filter using the Yang algorithm.";
        };

        static void initInputs(dv::InputDefinitionList &in) {
            in.addEventInput("events");
        };

        static void initOutputs(dv::OutputDefinitionList &out) {
            out.addEventOutput("events");
        };

        static void initConfigOptions(dv::RuntimeConfig &config) {
            config.add("deltaT", dv::ConfigOption::intOption("Time delta for filter.", 1000, 1, 1000000));
            config.add("threshold", dv::ConfigOption::intOption("Number of supporting pixels.", 1, 0, 10));

            config.setPriorityOptions({"deltaT", "threshold"});
        };

        KhodamoradiNoise() {
            sizeX    = inputs.getEventInput("events").sizeX();
            sizeY    = inputs.getEventInput("events").sizeY();
            _LENGTH_ = sizeX * sizeY;
            outputs.getEventOutput("events").setup(inputs.getEventInput("events"));
        };

        ~KhodamoradiNoise() {}

        void run() override {
            auto inEvent  = inputs.getEventInput("events").events();
            auto outEvent = outputs.getEventOutput("events").events();

            if (!inEvent) {
                return;
            }

            for (auto &evt : inEvent) {
                bool isNoise = calculateDensity(evt.x(), evt.y(), evt.timestamp(), evt.polarity());

                if (isNoise) {
                    outEvent << evt;
                }
            }
            outEvent << dv::commit;
        };

        void configUpdate() override {
            deltaT    = config.getInt("deltaT");
            threshold = config.getInt("threshold");
            regenerateParam();
        };
    };

}

registerModuleClass(kdv::KhodamoradiNoise)
