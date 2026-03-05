/**
 * BSD-3 LICENSE
 * Copyright (c) 2020-2025 by Bernd Porr
 * Copyright (c) 2020-2022 by Sama Daryanavard
 **/

#ifndef _DNF_H
#define _DNF_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <torch/torch.h>
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
class DNF
{
public:
    /**
     * Options for activation functions of all neurons in the network.
     **/
    enum ActMethod
    {
        Act_Sigmoid = 1,
        Act_Tanh = 2,
        Act_ReLU = 3,
        Act_NONE = 0
    };

private:
    struct Net : public torch::nn::Module
    {
        std::vector<torch::nn::Linear> fc;
        Net(int nLayers, int nInput, bool withBias = false);
        torch::Tensor forward(torch::Tensor x, ActMethod am);
    };

public:
    /**
     * Constructor which sets up the delay lines, network layers
     * and also calculates the number of neurons per layer so
     * that the final layer always just has one neuron.
     * \param nLayers Number of layers
     * \param nTaps Number of taps for the delay line feeding into the 1st layer
     * \param am The activation function for the neurons. Default is tanh.
     * \param tryGPU Does the learning on the GPU if available.
     **/
    DNF(const int nLayers,
        const int nTaps,
        const ActMethod am = Act_Tanh,
        const bool tryGPU = false);

    /**
     * Sets the learning rate of the entire network. It can
     * be set any time during learning. Setting it to zero
     * disables learning / adaptation.
     * \param mu Learning rate
     **/
    void setLearningRate(float mu);

    /**
     * Realtime sample by sample filtering operation
     * \param signal The signal contaminated with noise. Should be less than one.
     * \param noise The reference noise. Should be less than one.
     * \returns The filtered signal where the noise has been removed by the DNF.
     **/
    float filter(const float signal, const float noise);

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
     * Gets the weight distances per layer
     * \returns The Eucledian weight distance in relation to the initial weights.
     **/
    const std::vector<float> getLayerWeightDistances() const;

    /**
     * Gets the overall weight distsance
     * \returns The sum of all layer weight distances.
     **/
    float getWeightDistance() const;

    /**
     * Gets the torch device for example to determine if
     * the GPU is being used.
     **/
    const torch::Device getTorchDevice() const
    {
        return device;
    }

    /**
     * Gets the torch model, for example, to read out the weights.
     **/
    const Net getModel() const
    {
        return model;
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

    void saveInitialParameters()
    {
        for (const auto &p : model.parameters())
        {
            initialParameters.push_back(p.detach().clone());
        }
    }

    const int noiseDelayLineLength;
    const int signalDelayLineLength;
    const ActMethod actMethod;
    Net model;
    torch::optim::SGD optimizer;
    std::vector<torch::Tensor> initialParameters;
    DelayLine signal_delayLine;
    DelayLine noise_delayLine;
    float remover = 0;
    float f_nn = 0;
    torch::Device device = torch::kCPU;
};

#endif
