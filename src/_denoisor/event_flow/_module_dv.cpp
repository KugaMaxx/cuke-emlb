#include "event_flow.hpp"

namespace kdv {

    class EventFlow : public edn::EventFlow, public dv::ModuleBase {
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
            config.add("deltaT", dv::ConfigOption::intOption("Time delta for filter.", 3000, 1, 10000));
            config.add("squareR", dv::ConfigOption::intOption("Radius R of matrix for density Matrix.", 1, 1, 7));
            config.add("threshold", dv::ConfigOption::floatOption("Threshold of velocity.", 2, 0, 10));

            config.setPriorityOptions({"deltaT", "squareR", "threshold"});
        };

        EventFlow() {
            sizeX    = inputs.getEventInput("events").sizeX();
            sizeY    = inputs.getEventInput("events").sizeY();
            _LENGTH_ = sizeX * sizeY;
            outputs.getEventOutput("events").setup(inputs.getEventInput("events"));
        };

        ~EventFlow() {}

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
            squareR   = config.getInt("squareR");
            threshold = config.getFloat("threshold");
        };
    };

}

registerModuleClass(kdv::EventFlow)
