/**
 * BSD LICENSE
 * Copyright (c) 2020-2026 by Bernd Porr
 * Copyright (c) 2020-2022 by Sama Daryanavard
 **/

#ifndef _DNF_EXECUTORCH_H
#define _DNF_EXECUTORCH_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/extension/training/module/training_module.h>
#include <executorch/extension/training/optimizer/sgd.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <thread>
#include <iostream>
#include <deque>

#ifdef NDEBUG
constexpr bool debugOutput = false;
#else
constexpr bool debugOutput = true;
#endif

/**
 * Deep Neuronal Filter
 * https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0277974
 **/
class DNF_executorch
{
public:
    DNF_executorch(std::string features_pte_filename = "dnf_executorch.pte", double learningRate = 0.0001)
    {
        // Load the model file.
        executorch::runtime::Result<executorch::extension::FileDataLoader>
            loader_res =
                executorch::extension::FileDataLoader::from(features_pte_filename.c_str());
        if (loader_res.error() != executorch::runtime::Error::Ok)
        {
            fprintf(stderr, "Failed to open model file: %s", features_pte_filename.c_str());
            throw executorch::runtime::Error(loader_res.error());
        }
        auto loader = std::make_unique<executorch::extension::FileDataLoader>(
            std::move(loader_res.get()));

        std::unique_ptr<executorch::extension::FileDataLoader> ptd_loader = nullptr;

        trainingNet = std::make_shared<executorch::extension::training::TrainingModule>(
            std::move(loader), nullptr, nullptr, nullptr, std::move(ptd_loader));

        const auto method_meta = trainingNet->method_meta("forward");
        if (!method_meta.ok())
            throw method_meta.error();

        const auto input0_meta = method_meta->input_tensor_meta(0);
        noiseDelayLineLength = input0_meta->sizes()[1];
        noise_delayLine.init(noiseDelayLineLength);
        if (debugOutput)
            fprintf(stderr, "Noisedelayline length = %d\n", noiseDelayLineLength);
        noiseTimeSeries = executorch::extension::zeros({1, noiseDelayLineLength});

        signalDelayLineLength = noiseDelayLineLength / 2;
        signal_delayLine.init(signalDelayLineLength);
        delayedSignalTensor = executorch::extension::make_tensor_ptr<float>({1});
        if (debugOutput)
            fprintf(stderr, "Signaldelayline length = %d\n", signalDelayLineLength);

        if (debugOutput)
        {
            std::cerr << std::endl;
            const int nInputs = (int)(method_meta->num_inputs());
            std::cerr << "Num of inputs: " << nInputs << std::endl;
            for (int i = 0; i < nInputs; i++)
            {
                std::cerr << "Input " << i << ":" << std::endl;
                const auto input_meta = method_meta->input_tensor_meta(i);
                if (input_meta.ok())
                {
                    std::cerr << "Input Scalar type: " << type_to_string(input_meta->scalar_type()) << std::endl;
                    std::cerr << "Sizes: ";
                    for (auto &s : input_meta->sizes())
                        std::cerr << s << " ";
                    std::cerr << std::endl;
                }
            }

            std::cerr << std::endl;
            const int nOutputs = (int)(method_meta->num_outputs());
            std::cerr << "Num of outputs: " << nOutputs << std::endl;
            for (int i = 0; i < nOutputs; i++)
            {
                const auto output_meta = method_meta->output_tensor_meta(i);
                if (output_meta.ok())
                {
                    std::cerr << "Output #" << i << ":" << std::endl;
                    std::cerr << "Output Scalar type: " << type_to_string(output_meta->scalar_type()) << std::endl;
                    std::cerr << "Sizes: ";
                    for (auto &s : output_meta->sizes())
                        std::cerr << s << " ";
                    std::cerr << std::endl;
                }
            }

            auto param = trainingNet->named_parameters("forward").get();
            for (const auto &p : param)
            {
                std::cerr << "Param: " << p.first << " " << p.second.numel() << " weights." << std::endl;
            }
        }

        executorch::runtime::Error result = trainingNet->load_forward();
        if (executorch::runtime::Error::Ok != result)
        {
            std::cerr << "load_forward() error = " << (int)result << ", " << executorch::runtime::to_string(result) << std::endl;
            throw result;
        }

        executorch::runtime::Error error = trainingNet->load();
        if (!(trainingNet->is_loaded()))
        {
            std::cerr << "load() error = " << (int)error << executorch::runtime::to_string(error) << std::endl;
            throw error;
        }

        // Create optimizer.
        auto param_res = trainingNet->named_parameters("forward");
        if (param_res.error() != executorch::runtime::Error::Ok)
        {
            std::cerr << "getting parameters error = " << (int)param_res.error()
                      << executorch::runtime::to_string(param_res.error()) << std::endl;
            throw executorch::runtime::Error(param_res.error());
        }

        executorch::extension::training::optimizer::SGDOptions options{learningRate};
        optimizer = std::make_shared<executorch::extension::training::optimizer::SGD>(param_res.get(), options);
    }

    /**
     * Realtime sample by sample filtering operation
     * \param signal The signal contaminated with noise. Should be less than one.
     * \param noise The reference noise. Should be less than one.
     * \returns The filtered signal where the noise has been removed by the DNF.
     **/
    float filter(const float signal, const float noise)
    {
        const float delayed_signal = signal_delayLine.process(signal);
        delayedSignalTensor->mutable_data_ptr<float>()[0] = delayed_signal;

        noise_delayLine.process(noise);

        for (int i = 0; i < noiseDelayLineLength; i++)
        {
            noiseTimeSeries->mutable_data_ptr<float>()[i] = noise_delayLine.get(i);
        }

        const auto &results = trainingNet->execute_forward_backward("forward", {noiseTimeSeries, delayedSignalTensor});

        if (results.error() != executorch::runtime::Error::Ok)
        {
            fprintf(stderr, "Failed to execute forward_backward");
            return 0;
        }
        if (learningIsOn)
        {
            optimizer->step(trainingNet->named_gradients("forward").get());
        }

        remover = results.get()[1].toTensor().const_data_ptr<float>()[0];
        f_nn = delayed_signal - remover;

        return f_nn;
    }

    /**
     * Switches learning on or off.
     * @param If true learning is switched on. Otherwise off.
     */
    void setLearning(bool wants2learn = true)
    {
        learningIsOn = wants2learn;
    }

    /**
     * Get the Number Of Taps feeding into the deep net.
     * 
     * The number of taps is defining by the PTE file and are set when loading it.
     * With this function one can obtain the number of taps of the delay line feeding
     * into the Deep Network.
     * \returns Number of taps
     */
    inline int getNumberOfTaps() const
    {
        return noiseDelayLineLength;
    }

    /**
     * Returns the length of the delay line which
     * delays the signal polluted with noise.
     * \returns Number of delay steps in samples.
     **/
    inline int getSignalDelaySteps() const
    {
        return signalDelayLineLength;
    }

    /**
     * Returns the delayed with noise polluted signal by the delay
     * indicated by getSignalDelaySteps().
     * \returns The delayed noise polluted signal sample.
     **/
    inline float getDelayedSignal() const
    {
        return signal_delayLine.get(0);
    }

    /**
     * Returns the remover signal.
     * \returns The current remover signal sample.
     **/
    inline float getRemover() const
    {
        return remover;
    }

    /**
     * Returns the output of the DNF: the noise
     * free signal.
     * \returns The current output of the DNF which is idential to filter().
     **/
    inline float getOutput() const
    {
        return f_nn;
    }

    /**
     * Xavier gain for the weight init.
     **/
    static constexpr double xavierGain = 0.01;

private:
    class DelayLine
    {
    public:
        void init(int delay)
        {
            delaySamples = delay;
            buffer = std::deque<float>(delaySamples, 0.0f);
        }

        inline float process(float input)
        {
            float output = buffer.front();
            buffer.pop_front();
            buffer.push_back(input);
            return output;
        }

        float get(int i) const
        {
            return buffer[i];
        }

        float getNewest() const
        {
            return buffer.back();
        }

    private:
        int delaySamples = 0;
        std::deque<float> buffer;
    };

    int noiseDelayLineLength = 0;
    int signalDelayLineLength = 0;
    DelayLine signal_delayLine;
    DelayLine noise_delayLine;
    float remover = 0;
    float f_nn = 0;
    std::shared_ptr<executorch::extension::training::TrainingModule> trainingNet;
    std::shared_ptr<executorch::extension::training::optimizer::SGD> optimizer;
    bool learningIsOn = true;
    executorch::extension::TensorPtr noiseTimeSeries;
    executorch::extension::TensorPtr delayedSignalTensor;

    inline const char *type_to_string(executorch::aten::ScalarType t)
    {
        switch (t)
        {
        case executorch::aten::ScalarType::Byte:
            return "Byte";
        case executorch::aten::ScalarType::Char:
            return "Char";
        case executorch::aten::ScalarType::Short:
            return "Short";
        case executorch::aten::ScalarType::Int:
            return "Int";
        case executorch::aten::ScalarType::Long:
            return "Long";
        case executorch::aten::ScalarType::Float:
            return "Float";
        case executorch::aten::ScalarType::Double:
            return "Double";
        default:
            return "Unknown";
        }
    }
};

#endif
