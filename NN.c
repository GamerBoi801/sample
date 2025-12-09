#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Define the structure of the neural network
typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    double *hidden_layer;
    double *output_layer;
    double *input_layer;
    double *hidden_weights;
    double *output_weights;
    double *hidden_bias;
    double *output_bias;
} NeuralNetwork;

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid for backpropagation
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Initialize the neural network
NeuralNetwork* create_neural_network(int input_size, int hidden_size, int output_size) {
    NeuralNetwork *nn = (NeuralNetwork *) malloc(sizeof(NeuralNetwork));

    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;

    // Allocate memory for the layers and weights
    nn->input_layer = (double *) malloc(input_size * sizeof(double));
    nn->hidden_layer = (double *) malloc(hidden_size * sizeof(double));
    nn->output_layer = (double *) malloc(output_size * sizeof(double));

    nn->hidden_weights = (double *) malloc(input_size * hidden_size * sizeof(double));
    nn->output_weights = (double *) malloc(hidden_size * output_size * sizeof(double));

    nn->hidden_bias = (double *) malloc(hidden_size * sizeof(double));
    nn->output_bias = (double *) malloc(output_size * sizeof(double));

    // Initialize weights and biases with small random values
    srand(time(NULL));
    for (int i = 0; i < input_size * hidden_size; i++) {
        nn->hidden_weights[i] = (rand() % 2000 - 1000) / 1000.0;
    }

    for (int i = 0; i < hidden_size * output_size; i++) {
        nn->output_weights[i] = (rand() % 2000 - 1000) / 1000.0;
    }

    for (int i = 0; i < hidden_size; i++) {
        nn->hidden_bias[i] = (rand() % 2000 - 1000) / 1000.0;
    }

    for (int i = 0; i < output_size; i++) {
        nn->output_bias[i] = (rand() % 2000 - 1000) / 1000.0;
    }

    return nn;
}

// Forward propagation
void forward_propagate(NeuralNetwork *nn) {
    // Hidden layer activations
    for (int i = 0; i < nn->hidden_size; i++) {
        nn->hidden_layer[i] = 0;
        for (int j = 0; j < nn->input_size; j++) {
            nn->hidden_layer[i] += nn->input_layer[j] * nn->hidden_weights[i * nn->input_size + j];
        }
        nn->hidden_layer[i] += nn->hidden_bias[i];
        nn->hidden_layer[i] = sigmoid(nn->hidden_layer[i]);
    }

    // Output layer activations
    for (int i = 0; i < nn->output_size; i++) {
        nn->output_layer[i] = 0;
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->output_layer[i] += nn->hidden_layer[j] * nn->output_weights[i * nn->hidden_size + j];
        }
        nn->output_layer[i] += nn->output_bias[i];
        nn->output_layer[i] = sigmoid(nn->output_layer[i]);
    }
}

// Compute loss (Mean Squared Error)
double compute_loss(NeuralNetwork *nn, double *target) {
    double loss = 0.0;
    for (int i = 0; i < nn->output_size; i++) {
        loss += pow(nn->output_layer[i] - target[i], 2);
    }
    return loss / 2.0;
}

// Backpropagation to update weights
void backpropagate(NeuralNetwork *nn, double *target, double learning_rate) {
    // Output layer error and delta
    double *output_errors = (double *) malloc(nn->output_size * sizeof(double));
    double *output_deltas = (double *) malloc(nn->output_size * sizeof(double));

    for (int i = 0; i < nn->output_size; i++) {
        output_errors[i] = target[i] - nn->output_layer[i];
        output_deltas[i] = output_errors[i] * sigmoid_derivative(nn->output_layer[i]);
    }

    // Hidden layer error and delta
    double *hidden_errors = (double *) malloc(nn->hidden_size * sizeof(double));
    double *hidden_deltas = (double *) malloc(nn->hidden_size * sizeof(double));

    for (int i = 0; i < nn->hidden_size; i++) {
        hidden_errors[i] = 0;
        for (int j = 0; j < nn->output_size; j++) {
            hidden_errors[i] += output_deltas[j] * nn->output_weights[j * nn->hidden_size + i];
        }
        hidden_deltas[i] = hidden_errors[i] * sigmoid_derivative(nn->hidden_layer[i]);
    }

    // Update output weights and biases
    for (int i = 0; i < nn->output_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->output_weights[i * nn->hidden_size + j] += learning_rate * output_deltas[i] * nn->hidden_layer[j];
        }
        nn->output_bias[i] += learning_rate * output_deltas[i];
    }

    // Update hidden weights and biases
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->input_size; j++) {
            nn->hidden_weights[i * nn->input_size + j] += learning_rate * hidden_deltas[i] * nn->input_layer[j];
        }
        nn->hidden_bias[i] += learning_rate * hidden_deltas[i];
    }

    free(output_errors);
    free(output_deltas);
    free(hidden_errors);
    free(hidden_deltas);
}

// Train the neural network
void train(NeuralNetwork *nn, double **input_data, double **target_data, int data_size, int epochs, double learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0.0;
        for (int i = 0; i < data_size; i++) {
            // Set the input data
            for (int j = 0; j < nn->input_size; j++) {
                nn->input_layer[j] = input_data[i][j];
            }

            // Perform forward propagation
            forward_propagate(nn);

            // Compute loss for this sample
            epoch_loss += compute_loss(nn, target_data[i]);

            // Perform backpropagation
            backpropagate(nn, target_data[i], learning_rate);
        }
        epoch_loss /= data_size;
        printf("Epoch %d, Loss: %f\n", epoch + 1, epoch_loss);
    }
}

int main() {
    // Example data (XOR problem)
    double input_data[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    double target_data[4][1] = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // Convert to pointers for easier passing to functions
    double *input_ptr[4] = {input_data[0], input_data[1], input_data[2], input_data[3]};
    double *target_ptr[4] = {target_data[0], target_data[1], target_data[2], target_data[3]};

    // Create the neural network (2 input, 2 hidden, 1 output)
    NeuralNetwork *nn = create_neural_network(2, 2, 1);

    // Train the network
    train(nn, input_ptr, target_ptr, 4, 10000, 0.1);

    // Test the network
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < nn->input_size; j++) {
            nn->input_layer[j] = input_data[i][j];
        }
        forward_propagate(nn);
        printf("Input: (%.1f, %.1f), Predicted Output: %.5f\n",
               input_data[i][0], input_data[i][1], nn->output_layer[0]);
    }

    // Clean up
    free(nn->input_layer);
    free(nn->hidden_layer);
    free(nn->output_layer);
    free(nn->hidden_weights);
    free(nn->output_weights);
    free(nn->hidden_bias);
    free(nn->output_bias);
    free(nn);

    return 0;
}
