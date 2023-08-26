
import numpy as np

# ---------------------
# Data Generation
# ---------------------

np.random.seed(42)
features = np.random.rand(10000, 5)
weights_true = np.array([2, 3.5, -1.5, 2.1, -0.9])
bias_true = 4.2
noise = np.random.randn(10000) * 0.1
prices = features.dot(weights_true) + bias_true + noise

# Splitting data into training and validation sets
train_features = features[:8000]
train_prices = prices[:8000]
validation_features = features[8000:]
validation_prices = prices[8000:]

# ---------------------
# Model Initialization
# ---------------------

# Basic Linear Regression Training Function
def train_linear_regression(features, targets, epochs, lr=0.01):
    num_features = features.shape[1]
    weights = np.random.randn(num_features)
    bias = 0
    
    for epoch in range(epochs):
        predictions = features.dot(weights) + bias
        errors = predictions - targets
        weights_gradient = 2/len(features) * features.T.dot(errors)
        bias_gradient = 2 * np.mean(errors)
        
        weights -= lr * weights_gradient
        bias -= lr * bias_gradient
        
    return weights, bias

# ---------------------
# Simultaneous Epoch Training
# ---------------------

# Partitioned Data Approach
weights1, bias1 = train_linear_regression(train_features[:2667], train_prices[:2667], epochs=1)
weights2, bias2 = train_linear_regression(train_features[2667:5334], train_prices[2667:5334], epochs=1)
weights3, bias3 = train_linear_regression(train_features[5334:], train_prices[5334:], epochs=1)

# Same Data Approach
weights1_same_data, bias1_same_data = train_linear_regression(train_features, train_prices, epochs=1)
weights2_same_data, bias2_same_data = train_linear_regression(train_features, train_prices, epochs=1)
weights3_same_data, bias3_same_data = train_linear_regression(train_features, train_prices, epochs=1)

# ---------------------
# Weight Aggregation
# ---------------------

average_weights = np.mean([weights1, weights2, weights3], axis=0)
average_bias = np.mean([bias1, bias2, bias3])

average_weights_same_data = np.mean([weights1_same_data, weights2_same_data, weights3_same_data], axis=0)
average_bias_same_data = np.mean([bias1_same_data, bias2_same_data, bias3_same_data])

# ---------------------
# Evaluation
# ---------------------

def evaluate_linear_regression(features, weights, bias, true_values):
    predictions = features.dot(weights) + bias
    mse = np.mean((predictions - true_values) ** 2)
    return mse

mse_simultaneous_partitioned = evaluate_linear_regression(validation_features, average_weights, average_bias, validation_prices)
mse_simultaneous_same_data = evaluate_linear_regression(validation_features, average_weights_same_data, average_bias_same_data, validation_prices)
weights_traditional, bias_traditional = train_linear_regression(train_features, train_prices, epochs=1)
mse_traditional = evaluate_linear_regression(validation_features, weights_traditional, bias_traditional, validation_prices)
