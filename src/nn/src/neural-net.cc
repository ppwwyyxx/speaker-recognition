/*
 * $File: neural-net.cc
 * $Date: Sun Nov 17 12:51:11 2013 +0000
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#include "neural-net.hh"
#include "activation.hh"

#include "Threadpool/Threadpool.hpp"

#include <cassert>
#include <limits>

using namespace std;

static real_t net_output(std::vector<real_t> &weights, std::vector<real_t> &input) {
	// weights has one more value than output, a.k.a bias
	assert(input.size() + 1 == weights.size());

	real_t net = weights[0]; // bias first

	for (size_t i = 0; i < input.size(); i ++)
		net += input[i] * weights[i + 1];
	return net;
}

static void svm2vec(const std::vector<std::pair<int, real_t>> &input,
		std::vector<real_t> &output, int dim) {
	output = vector<real_t>(dim, 0);
	for (auto &item: input) {
		if (item.first > dim) // ignore out bounded values
			continue;
		output[item.first] = item.second;
	}
}

Neuron::Neuron() {
	random.set_normal_distribution_param(0, 0.01);
}

void Neuron::reset_weights() {
	for (auto &weight: weights) {
		//weight = random.rand_real() * (random.rand_int(2) * 2 - 1);
		weight = random.rand_normal();
	}
}

void NeuralNetLayer::activate(std::vector<real_t> &x, std::vector<real_t> &activation) {
	activation.resize(neurons.size());
	int i = 0;
	for (auto &neuron: neurons) {
		neuron.net = net_output(neuron.weights, x);
		neuron.activation = activation_function->value_at(neuron.net);
		activation[i ++] = neuron.activation;
	}
}

NeuralNetLayer::NeuralNetLayer(int n_neuron) {
	neurons.resize(n_neuron);
	activation_function = make_shared<ActivationLogistic>();
}

//NeuralNetLayer::NeuralNetLayer(const NeuralNetLayer &nnl) {
//    neurons = nnl.neurons;
//    activation_function = nnl.activation_function;
//}

NeuralNetLayer::~NeuralNetLayer() {
}

NeuralNet::NeuralNet(int n_iter, real_t learning_rate) {
	this->n_iter = n_iter;
	this->learning_rate = learning_rate;
	this->has_rbm_layer = false;
}

void NeuralNet::add_layer(int n_neuron) {
	if (!has_rbm_layer)
		layer_neuron_count.push_back(n_neuron);
	else {
		int nweights = layers.back().n_neurons() + 1;
		layers.emplace_back(NeuralNetLayer(n_neuron));
		layers.back().set_nweights(nweights);
	}
}

void NeuralNet::from_dbn(DBN &dbn) {
	has_rbm_layer = true;
	auto &rbms = dbn.rbms;
	assert(rbms.size() > 0);
	layers.resize(0);
	for (size_t i = 0; i < rbms.size(); i ++) {
		auto &rbm = rbms[i];
		layers.emplace_back(NeuralNetLayer(rbm->get_hidden_layer_size()));
		auto &layer = layers.back();
		layer.set_nweights(rbm->get_visible_layer_size() + 1);
		assert((int)layer.neurons.size() == rbm->get_hidden_layer_size());
		for (size_t j = 0; j < layer.neurons.size(); j ++) {
			auto &neuron = layer.neurons[j];
			assert((int)neuron.weights.size() == rbm->get_visible_layer_size() + 1);
			neuron.weights[0] = rbm->hidden_layer_bias[j];
			for (size_t k = 0; k < neuron.weights.size() - 1; k ++)
				neuron.weights[k + 1] = rbm->w[k][j];
		}
		if (i > 0)
			assert(layers[i].neurons[0].weights.size() == layers[i - 1].neurons.size() + 1);
	}
}

void NeuralNet::reset_NN(std::pair<int, int> dimension, int n_out_category) {
	layers.resize(0);
	int n_prev_layer_output = dimension.second;
	for (auto &cnt: layer_neuron_count) {
		layers.push_back(NeuralNetLayer(cnt));
		layers.back().set_nweights(n_prev_layer_output + 1);
		n_prev_layer_output = cnt;
	}
	layers.push_back(NeuralNetLayer(n_out_category));
	layers.back().set_nweights(n_prev_layer_output + 1);
	reset_weights();
}

void NeuralNet::forward_propagate(std::vector<real_t> &x) {
	vector<real_t> activation[2];
	first_layer().activate(x, activation[0]);
	for (size_t i = 1; i < layers.size(); i ++)
		// switch according to parity
		layers[i].activate(activation[(i + 1) & 1], activation[i & 1]);
}

void NeuralNet::back_propagate(std::vector<real_t> &y) {

	// delta of output layer
	int i = 0;
	for (auto &neuron: output_layer().neurons) {
		// TODO: this may be other metrics such as squared error
		real_t error_signal = (y[i ++] - neuron.activation);

		real_t derivative =
			output_layer().activation_function->derivative(neuron.net, neuron.activation);

		//printf("x:%f %f %f\n", error_signal, derivative, neuron.activation);
		neuron.delta = error_signal * derivative;
	}

	// delta of other layers
	for (int i = ((int)layers.size()) - 1; i >= 1; i --) {
		NeuralNetLayer &current_layer = layers[i],
					   &previous_layer = layers[i - 1];
		for (size_t j = 0; j < previous_layer.neurons.size(); j ++) {
			auto &neuron = previous_layer.neurons[j];
			neuron.delta = 0;
			for (auto &next_neuron: current_layer.neurons) {
				assert(j < next_neuron.weights.size());
				neuron.delta += next_neuron.weights[j] * next_neuron.delta;
			}
			neuron.delta *= previous_layer.activation_function->derivative(
					neuron.net, neuron.activation);
		}
	}
}

void NeuralNet::update_weights(std::vector<real_t> &input) {
	// first layer
	for (size_t i = 0; i < first_layer().neurons.size(); i ++) {
		auto &neuron = first_layer().neurons[i];
		// biae
		real_t coef = learning_rate * neuron.delta;
		neuron.weights[0] += coef;
		// other weights
		for (size_t j = 1; j < neuron.weights.size(); j ++) {
			real_t delta_weight = coef * input[j - 1];
			//printf("%f %f\n", neuron.delta, delta_weight);
			neuron.weights[j] += delta_weight;
		}
	}

	//NeuralNetLayer prev_layer = first_layer();
	// following layers
	for (size_t i = 1; i < layers.size(); i ++) {
		NeuralNetLayer &cur_layer = layers[i],
					   &prev_layer = layers[i - 1];
		for (auto &neuron: cur_layer.neurons) {
			// update bias
			neuron.weights[0] += learning_rate * neuron.delta;
			// other weights
			for (size_t j = 1; j < neuron.weights.size(); j ++) {
				real_t delta_weight = learning_rate * neuron.delta * prev_layer.neurons[j - 1].activation;
				neuron.weights[j] += delta_weight;
			}
		}
	}
}

void NeuralNet::fit_incremental(std::vector<std::pair<int, real_t>> &x_svm, std::vector<real_t> &y) {
	assert(y.size() == output_layer().neurons.size()); // dimension must match

	vector<real_t> x;
	svm2vec(x_svm, x, first_layer().neurons[0].weights.size() - 1);

	// first pass: fill in activation
	forward_propagate(x);

	// backpropagate error
	back_propagate(y);

	// update weights
	update_weights(x);
}

void NeuralNet::reset_weights() {
	for (auto &layer: layers)
		for (auto &neuron: layer.neurons)
			neuron.reset_weights();
}

#if 0
void NeuralNet::fit(std::vector<std::vector<std::pair<int, real_t>>> &X,
		std::vector<real_t> &y) {
	throw "Sparse training not implemented.";
}

void NeuralNet::fit(std::vector<std::vector<real_t>> &X,
		std::vector<real_t> &y) {
	// training using Back Propagation algorithm
	assert(X.size() == y.size());

	// 1. initialize
	reset_weights();

	// 2. fit every input instance
	for (size_t i = 0; i < X.size(); i ++) {
		vector<real_t> desired_output(output_layer().neurons.size(), 0);
		int t = y[i];
		assert(t >= 0 && t < (int)desired_output.size());
		desired_output[t] = 1.0;
		fit_incremental(X[i], desired_output);
	}
}

#endif

void NeuralNet::predict(std::vector<std::vector<real_t>> &X,
		std::vector<real_t> &y) {
	y.resize(0);
	for (auto &x: X)
		y.push_back(predict_one(x));
}

void NeuralNet::predict(std::vector<std::vector<std::pair<int, real_t>>> &X,
		std::vector<real_t> &y) {
	y.resize(0);
	vector<real_t> x_dense;
	for (auto &x: X) {
		svm2vec(x, x_dense, first_layer().neurons[0].weights.size() - 1);
		y.push_back(predict_one(x_dense));
	}
}

real_t NeuralNet::predict_one(std::vector<std::pair<int, real_t>> &x_svm) {
	std::vector<real_t> x;
	svm2vec(x_svm, x, first_layer().neurons[0].weights.size() - 1);
	return predict_one(x);
}

real_t NeuralNet::predict_one(std::vector<real_t> &x) {
	forward_propagate(x);
	real_t max_activation = -numeric_limits<real_t>::max();
	real_t output = 0;
	for (size_t i = 0; i < output_layer().neurons.size(); i ++) {
		auto &neuron = output_layer().neurons[i];
		if (neuron.activation > max_activation) {
			max_activation = neuron.activation;
			output = i;
		}
	}
	return output;
}

/**
 * vim: syntax=cpp11 foldmethod=marker
 */
