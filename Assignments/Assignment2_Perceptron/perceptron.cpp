#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

// Define the Perceptron structure
struct Perceptron {
    vector<float> weights; // Weights
    float bias;            // Bias term
};

// Function to initialize perceptron weights and bias randomly
void initialize_perceptron(Perceptron &perceptron, int num_features) {
    srand(static_cast<unsigned>(time(0))); // random number generation
    perceptron.weights.resize(num_features);

    // Random initialization of weights
    for (float &weight : perceptron.weights) {
        weight = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
    }

    // Random initialization of bias
    perceptron.bias = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
}

// Activation function (step function)
int activation_function(float z) {
    return z >= 0 ? 1 : 0;
}

// Function to calculate the dot product
float dot_product(const vector<float> &vec1, const vector<float> &vec2) {
    float result = 0.0f;
    for (size_t i = 0; i < vec1.size(); ++i) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

// Perceptron training function
void train_perceptron(Perceptron &perceptron, const vector<vector<float>> &X, const vector<int> &Y) {
    int epochs = 0;
    bool converged = false;

    while (!converged) {
        converged = true; // Assume convergence until a misclassification is found
        ++epochs;

        for (size_t i = 0; i < X.size(); ++i) {
            // Compute weighted sum + bias
            float z = dot_product(perceptron.weights, X[i]) + perceptron.bias;
            int prediction = activation_function(z);
            int error = Y[i] - prediction;

            if (error != 0) {
                converged = false; // Mark as not converged if any misclassification occurs

                // Update weights and bias
                for (size_t j = 0; j < perceptron.weights.size(); ++j) {
                    perceptron.weights[j] += error * X[i][j]; // Adjust weights
                }
                perceptron.bias += error; // Adjust bias
            }
        }
    }
    cout << "Training complete after " << epochs << " epochs.\n";
}

// Main function to model AND gate with perceptron
int main() {
    // Input features (AND gate truth table inputs)
    vector<vector<float>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    
    // Output labels (AND gate truth table outputs)
    vector<int> Y = {0, 0, 0, 1};

    // Initialize perceptron with random weights and bias
    Perceptron perceptron;
    initialize_perceptron(perceptron, 2);

    // Display initial weights and bias
    cout << "Initial Weights: ";
    for (float weight : perceptron.weights) {
        cout << weight << " ";
    }
    cout << "\nInitial Bias: " << perceptron.bias << "\n";

    // Train the perceptron to model the AND gate
    train_perceptron(perceptron, X, Y);

    // Display the trained weights and bias
    cout << "Trained Weights: ";
    for (float weight : perceptron.weights) {
        cout << weight << " ";
    }
    cout << "\nTrained Bias: " << perceptron.bias << "\n";

    // Test the perceptron on all possible inputs
    cout << "Testing Perceptron:\n";
    for (size_t i = 0; i < X.size(); ++i) {
        float z = dot_product(perceptron.weights, X[i]) + perceptron.bias;
        int prediction = activation_function(z);
        cout << "Input: {" << X[i][0] << ", " << X[i][1] << "} => Prediction: " << prediction << "\n";
    }

    return 0;
}