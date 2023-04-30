#include "event_denoise_convolution_network.hpp"

namespace kdv {

    class EventDenoiseConvolutionNetwork : public edn::EventDenoiseConvolutionNetwork, public dv::ModuleBase {
    public:
        static const char *initDescription() {
            return "Noise filter using the EventDenoiseConvolutionNetwork.";
        };

        static void initInputs(dv::InputDefinitionList &in) {
            in.addEventInput("events");
        };

        static void initOutputs(dv::OutputDefinitionList &out) {
            out.addEventOutput("events");
        };

        static void initConfigOptions(dv::RuntimeConfig &config) {
            config.add("params", dv::ConfigOption::floatOption("Spatial blur coefficient.", 1.0, 0.1, 3.0));

            config.setPriorityOptions({"params"});
        };

        EventDenoiseConvolutionNetwork() {
            sizeX    = inputs.getEventInput("events").sizeX();
            sizeY    = inputs.getEventInput("events").sizeY();
            _LENGTH_ = sizeX * sizeY;
            outputs.getEventOutput("events").setup(inputs.getEventInput("events"));
        };

        ~EventDenoiseConvolutionNetwork() {}

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
            params = config.getFloat("params");
            regenerateParam();
        };
    };

}

registerModuleClass(kdv::EventDenoiseConvolutionNetwork)
