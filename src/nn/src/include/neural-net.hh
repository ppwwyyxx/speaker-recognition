/*
 * $File: neural-net.hh
 * $Date: Fri Nov 22 18:03:55 2013 +0000
 * $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>
 */

#pragma once
#include "type.hh"
#include "random.hh"
#include "activation.hh"

#include "rbm.hh"
#include "dbn.hh"

#include <vector>
#include <memory>

#include <string>
#include <sstream>
#include <cassert>

class Neuron {
	public:
		Neuron();
		/// Dimension of weights is the number of neuron in previous layer
		std::vector<real_t> weights;

		/// net output
		real_t net;

		/// temporal activation value, activation_function->value_at(net)
		real_t activation;

		/// local gradient
		real_t delta;

		/// initialize with small random numbers
		void reset_weights();

		/// set number of weights, bias included
		void set_nweights(int nweights) { weights.resize(nweights); }

	protected:
		/// random number generator
		Random random;
};

class NeuralNetLayer {
	public:
		NeuralNetLayer() {}
		NeuralNetLayer(int n_neuron);
//        NeuralNetLayer(const NeuralNetLayer &nnl);
		~NeuralNetLayer();

		/// neurons in this layer
		std::vector<Neuron> neurons;

		/// activation function, usually a logistic function.
		/// different layers may use different activation functions
		std::shared_ptr<ActivationFunction> activation_function;

		/// activate this layer according to output from previous layer
		/// @param x output from previous layer
		/// @param activation output activation value of this layer
		void activate(std::vector<real_t> &x, std::vector<real_t> &activation);

		/// number of neurons
		int n_neurons() const { return neurons.size(); }

		// set number of weights for all neurons in this layer
		inline void set_nweights(int nweights) {
			for (auto &neuron: neurons)
				neuron.set_nweights(nweights);
		}

};

class NeuralNet {
	public:

		NeuralNet(int n_iter = 100, real_t learning_rate = 0.1);

		/// add a layer of size @size.
		/// lazy operation: it will not enrich layer's information
		/// until training starts
		void add_layer(int size);

		void reset_NN(std::pair<int, int> dimension, int n_out_category);

		/// train Neural Net.
		// void fit(std::vector<std::vector<real_t>> &X, std::vector<real_t> &y);
		/// sparse version for fit: svm format
		// void fit(std::vector<std::vector<std::pair<int, real_t>>> &X, std::vector<real_t> &y);

		/// suppose there are K layers predefined, which means there are actually K + 1 layers
		/// in the network (one more output layer), this function will consider the first K layers
		/// as a Deep Belief Network, and train them unsupervisely and layer-pair-wisely,
		/// using Resrticted Boltzmann Machine model, with Gibbs Sampling for gradient estimation.
		/// After that, the whole network will be fine tune with Backpropagation algorithm.
		///
		/// IMPORTANT: from_dbn must be called first
		void fit_dbn(std::vector<std::vector<real_t>> &X, std::vector<real_t> &y,
				const std::vector<std::vector<real_t>> &X_test = std::vector<std::vector<real_t>>(),
				const std::vector<real_t> &y_test = std::vector<real_t>()) {
				do_fit(X, y, X_test, y_test, true);
		}

		void from_dbn(DBN &dbn);

		void print_dimension() {
			printf("n_layers: %d\n", (int)layers.size());
			for (auto &layer: layers)
				printf("%d ", (int)layer.neurons.size());
			printf("\n");
		}
		template<class InstType>
			void do_fit(std::vector<std::vector<InstType>> &X, std::vector<real_t> &y,
					const std::vector<std::vector<InstType>> &X_test = std::vector<std::vector<InstType>>(),
					const std::vector<real_t> &y_test = std::vector<real_t>(),
					bool continue_training = false) {
				printf("start training: niter_max: %d\n", n_iter); fflush(stdout);
				int max_y = std::numeric_limits<int>::min(),
					min_y = std::numeric_limits<int>::max();
				for (auto &yp: y) {
					max_y = std::max(max_y, (int)yp);
					min_y = std::min(min_y, (int)yp);
				}
				assert(X.size() == y.size());
				assert(X_test.size() == y_test.size());
				assert(min_y >= 0);

				if (!continue_training && !has_rbm_layer)
					reset_NN(get_data_dimension(X), max_y + 1);

				print_dimension();

				real_t learning_rate_save = this->learning_rate;
				real_t prev_error_rate = 1.0;
				int n_stagnation = 0;
				for (int iter_id = 0; iter_id < n_iter; iter_id ++) {
					for (size_t i = 0; i < X.size(); i ++) {
						std::vector<real_t> desired_output(
								output_layer().neurons.size(), 0.0);
						int t = y[i];
						assert(t >= 0 && t < (int)desired_output.size());
						desired_output[t] = 1.0;
						fit_incremental(X[i], desired_output);
					}

					// inspect deltas
					int n_neuron = 0;
					real_t sum = 0;
					NeuralNetLayer &layer = output_layer();
						for (auto &neuron: layer.neurons) {
							n_neuron ++;
							sum += fabs(layer.activation_function->derivative(neuron.net, neuron.activation));
						}

					//printf("ave : %f\n", sum / n_neuron);
					real_t error = 0;
					std::vector<real_t> preds(X.size());
					for (size_t i = 0; i < X.size(); i ++) {
						int y_pred = predict_one(X[i]);
						preds[i] = y_pred;
						int y_true = y[i];
						int e = y_pred != y_true;

						if (error == 0 && e != 0) {
							printf("%d:%d-%d: ", (int)i, (int)y[i], (int)y_pred);
							int j = 0;
							for (auto &neuron: output_layer().neurons) {
								if (j == y_pred || j == y_true)
									printf("%d:%f,%f ", j, neuron.activation, neuron.delta);
								j ++;
							}
							printf("\n");
						}
						error += e;
					}

					printf("------ on training data ------\n");
//                    stats(y, preds);

					preds.resize(X_test.size());
					real_t test_error = 0;
					for (size_t i = 0; i < X_test.size(); i ++) {
						auto x = X_test[i];
						int y_pred = predict_one(x);
						preds[i] = y_pred;
						int y_true = y_test[i];
						test_error += y_pred != y_true;
					}
					if (X_test.size() != 0)
						test_error /= X_test.size();

					printf("------ on test data ------\n");
//                    stats(y_test, preds);

					real_t training_error = error / (double)X.size();
					printf("[iter %d] %g %g\n", iter_id, training_error, test_error);
					if (training_error>= prev_error_rate) {
						n_stagnation ++;
						printf("stagnation %d: not improving performance.\n", n_stagnation);
					}
					else {
						n_stagnation = 0;
						prev_error_rate = training_error;
					}
					if (n_stagnation > 100) {
						printf("stagnated for %d times. \n", n_stagnation);
//                        printf("stagnated for %d times. stop training.\n", n_stagnation);
//                        break;
					}
//                    if (training_error < 0.01)
//                        this->learning_rate = 0.01;
					fflush(stdout);
				}
				this->learning_rate = learning_rate_save;
			}

		template<class InstType>
			void fit(std::vector<std::vector<InstType>> &X, std::vector<real_t> &y,
					const std::vector<std::vector<InstType>> &X_test = std::vector<std::vector<InstType>>(),
					const std::vector<real_t> &y_test = std::vector<real_t>()) {
				do_fit(X, y, X_test, y_test, false);
			}

		/// incrementally train the network with one instance
		/// y is the desired output of the last layer.
		/// when performing classification tasks, it may not
		/// not be feasible to used the label as the output
		/// of the last layer. some kind of encoding from
		/// the label to the output layer may be performed
		/// before calling this method
		void fit_incremental(std::vector<real_t> &x, std::vector<real_t> &y);
		/// sparse version of fit_incremental:
		void fit_incremental(std::vector<std::pair<int, real_t>> &x, std::vector<real_t> &y);

		/// classification single output
		void predict(std::vector<std::vector<real_t>> &X,
				std::vector<real_t> &y);
		/// sparse version for predict: svm format
		void predict(std::vector<std::vector<std::pair<int, real_t>>> &X,
				std::vector<real_t> &y);
		/// predict one instance for classification
		real_t predict_one(std::vector<real_t> &x);
		real_t predict_one(std::vector<std::pair<int, real_t>> &x);

		void set_learning_rate(real_t learning_rate) { this->learning_rate = learning_rate; }
		void set_niter_max(int n_iter) { this->n_iter= n_iter; }

	protected:
	public:
		int n_iter;

		std::vector<int> layer_neuron_count;

		real_t learning_rate;

		bool has_rbm_layer;

		// the last layer is output layer,
		// and its dimension must match the output dimension
		std::vector<NeuralNetLayer> layers;


		// generic option set method
//        template<class T>
//        void set(const std::string &optname,
//                T value) {
//        }

//        template<class From, class To>
//        To convert_value(const From &from) {
//            std::stringstream ss;
//            To to;
//            ss << from;
//            ss >> to;
//            return to;
//        }

		void reset_weights();
		/// perform forward propagation, fill in neuron activation
		void forward_propagate(std::vector<real_t> &x);
		/// back propagate
		void back_propagate(std::vector<real_t> &y);
		/// update weights
		void update_weights(std::vector<real_t> &input);

		/// return the first layer, which is adjacent to input
		NeuralNetLayer &first_layer() { return layers[0]; }
		/// return the output layer, which is the last layer
		NeuralNetLayer &output_layer() {
			assert(layers.size() != 0);
			return layers.back();
		}

		inline static std::pair<int, int> get_data_dimension(std::vector<std::vector<real_t>> &X) {
			if (X.size() == 0)
				return std::make_pair(0, 0);
			return std::make_pair((int)X.size(), X[0].size());
		}

		inline static std::pair<int, int> get_data_dimension(std::vector<std::vector<std::pair<int, real_t>>> &X) {
			int m = 0;
			for (auto &x: X)
				for (auto &item: x)
					if (item.first > m)
						m = item.first;
			return std::make_pair((int)X.size(), m + 1);
		}


		/// random number generator
		Random random;
		void stats(const std::vector<real_t> &y_true,
				const std::vector<real_t> &y_pred) {
			assert(y_true.size() == y_pred.size());
			int max_v = -1;
			for (auto &y: y_true)
				max_v = std::max(max_v, (int)y);
			int n_cls = max_v + 1;
			std::vector<std::pair<int, int>> stat(n_cls);
			std::vector<int> cls_cnt(n_cls);

			real_t acc = 0;
			for (size_t i = 0; i < y_true.size(); i ++) {
				int error = y_true[i] != y_pred[i];
				if (!error) {
					stat[y_true[i]].first ++;
					acc += 1.0;
				}
				else stat[y_true[i]].second ++;
				cls_cnt[y_pred[i]] ++;
			}
			acc /= y_true.size();

			real_t sp = 0, sr = 0, sf = 0;
			printf("lbl precis recall f1     acc\n");
			// class precision, recall, F1
			for (int i = 0; i < n_cls; i ++) {
				real_t p = 0, r;
				if (cls_cnt[i] != 0)
					p = stat[i].first / (real_t)cls_cnt[i];
				r = stat[i].first / (real_t)(stat[i].first + stat[i].second);
				real_t f = 0;
				if (p + r > 0)
					f = 2 * p * r / (p + r);
				printf("%3d %.4f %.4f %.4f\n", i, p, r, f);
				sp += p, sr += r, sf += f;
			}
			sp /= n_cls, sr /= n_cls, sf /= n_cls;
			printf("ave %.4f %.4f %.4f %.4f\n", sp, sr, sf, acc);
		}

};

/**
 * vim: syntax=cpp11 foldmethod=marker
 */

