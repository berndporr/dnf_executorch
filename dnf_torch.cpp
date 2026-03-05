#include "dnf_torch.h"

float DNF::filter(const float signal, const float noise)
{
	const float delayed_signal = signal_delayLine.process(signal);
	noise_delayLine.process(noise);

	auto noiseTimeSeries = executorch::extension::zeros({noiseDelayLineLength});
	for (int i = 0; i < noiseDelayLineLength; i++)
	{
		noiseTimeSeries->mutable_data_ptr<float>()[i] = noise_delayLine.get(i);
	}

    std::vector<executorch::aten::SizesType> insizes({noiseDelayLineLength});

	auto output = (trainingNet->forward({noiseTimeSeries,delayed_signal}));

	// torch::Tensor gradient = torch::tensor({-f_nn}).to(device);
	//output.retain_grad();
	// output.backward(gradient);

	return f_nn;
}

float DNF::getWeightDistance() const
{
	return 0;
}

const std::vector<float> DNF::getLayerWeightDistances() const
{
	std::vector<float> distances;
	return distances;
}

void DNF::setLearningRate(float mu)
{
}
