#include "time_surface.hpp"

namespace kdv {

    class TimeSurface : public edn::TimeSurface, public dv::ModuleBase {
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
            config.add("squareR", dv::ConfigOption::intOption("Radius R of matrix for time surface.", 1, 1, 7));
            config.add("threshold", dv::ConfigOption::floatOption("Threshold value for mean time surface value.", 0.3, 0, 1));

            config.setPriorityOptions({"decay", "squareR", "threshold"});
        };

        TimeSurface() {
            sizeX    = inputs.getEventInput("events").sizeX();
            sizeY    = inputs.getEventInput("events").sizeY();
            _LENGTH_ = sizeX * sizeY;
            outputs.getEventOutput("events").setup(inputs.getEventInput("events"));
        };

        ~TimeSurface() {}

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
            decay     = config.getInt("decay");
            squareR   = config.getInt("squareR");
            threshold = config.getFloat("threshold");
            regenerateParam();
        };
    };

}

registerModuleClass(kdv::TimeSurface)
