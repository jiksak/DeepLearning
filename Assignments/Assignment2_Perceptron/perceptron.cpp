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

void initialize_perceptron(Perceptron &perceptron, int num_features) {
    srand(static_cast<unsigned>(time(0))); // Random number generation
    perceptron.weights.resize(num_features);

    // Random initialization of weights
    for (float &weight : perceptron.weights) {
        weight = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) * 10); // Smaller random values
    }

    // Random initialization of bias
    perceptron.bias = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) * 10); // Smaller random value
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
    const int MAX_EPOCHS = 1000; // Prevent infinite loop
    bool converged = false;

    while (!converged && epochs < MAX_EPOCHS) {
        converged = true; // Assume convergence
        ++epochs;

        int misclassifications = 0; // Track errors in this epoch

        for (size_t i = 0; i < X.size(); ++i) {
            // Compute weighted sum + bias
            float z = dot_product(perceptron.weights, X[i]) + perceptron.bias;
            int prediction = activation_function(z);
            int error = Y[i] - prediction;

            if (error != 0) { // Misclassification
                converged = false; // Mark as not converged
                ++misclassifications;

                // Update weights: w <- w + x or w <- w - x
                if (error == 1) { // Prediction is 0 but should be 1
                    for (size_t j = 0; j < perceptron.weights.size(); ++j) {
                        perceptron.weights[j] += X[i][j]; // Add input to weights
                    }
                    perceptron.bias += 1; // Increase bias
                } else if (error == -1) { // Prediction is 1 but should be 0
                    for (size_t j = 0; j < perceptron.weights.size(); ++j) {
                        perceptron.weights[j] -= X[i][j]; // Subtract input from weights
                    }
                    perceptron.bias -= 1; // Decrease bias
                }
            }
        }

        // Debugging: Print weights, bias, and misclassifications for each epoch
        cout << "Epoch: " << epochs << ", Misclassifications: " << misclassifications << "\n";
    }

    if (epochs >= MAX_EPOCHS) {
        cout << "Training stopped after reaching the maximum number of epochs.\n";
    } else {
        cout << "Training complete after " << epochs << " epochs.\n";
    }
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
