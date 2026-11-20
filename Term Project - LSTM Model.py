# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

np.random.seed(42)

stock_name = "AAPL"
start_date = "2015-01-01"

window_size = 10
train_ratio = 0.8

hidden_size = 32
learning_rate = 0.01
total_loops = 20


# reading data
print("Downloading stock prices for", stock_name)

stock = yf.download(
    stock_name,
    start=start_date,
    auto_adjust=False,
    progress=False,
    multi_level_index=False
)

if stock.empty:
    raise ValueError("No stock data downloaded")


# get close price
if isinstance(stock.columns, pd.MultiIndex):
    if ("Close", stock_name) in stock.columns:
        closing_price = stock[("Close", stock_name)]
    elif (stock_name, "Close") in stock.columns:
        closing_price = stock[(stock_name, "Close")]
    else:
        raise ValueError("Cannot find close price")
else:
    closing_price = stock["Close"]

if closing_price.isna().sum() > 0:
    print("missing close values found")
    closing_price = closing_price.dropna()


print("Rows downloaded:", len(closing_price))
print("First trading day:", closing_price.index[0].date())
print("Last trading day:", closing_price.index[-1].date())


# first plot
plt.figure(figsize=(12, 6))
plt.plot(closing_price)
plt.title("AAPL Closing Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.tight_layout()
plt.close()


#data scaling
price_array = closing_price.values.reshape(-1, 1)
lowest_price = np.min(price_array)
highest_price = np.max(price_array)

if highest_price == lowest_price:
    raise ValueError("All close prices are same, cannot scale")

price_array = (price_array - lowest_price) / (highest_price - lowest_price)
scaled_prices = price_array


# creating sequence data
input_sequences = []
target_values = []

for index in range(len(scaled_prices.flatten()) - window_size):
    input_sequences.append(scaled_prices.flatten()[index:index + window_size])
    target_values.append(scaled_prices.flatten()[index + window_size])

input_data, target_output = np.array(input_sequences), np.array(target_values)


if len(input_data) == 0:
    raise ValueError("Not enough stock data to build sequences")

split = int(len(input_data) * train_ratio)

train_inputs = input_data[:split]
train_targets = target_output[:split]

test_inputs = input_data[split:]
test_targets = target_output[split:]

if len(train_inputs) == 0 or len(test_inputs) == 0:
    raise ValueError("Train or test set is empty")


print()
print("Project setup")
print("----------------------")
print("Ticker:", stock_name)
print("Window:", window_size)
print("Train size:", len(train_inputs))
print("Test size:", len(test_inputs))
print("Hidden units:", hidden_size)


# lstm model
class LSTM:
    def __init__(self, input_size=1, hidden_size=8, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        combined_size = input_size + hidden_size

        # setting up the four lstm gates
        self.gate_names = ["forget", "input", "memory", "output"]

        for gate in self.gate_names:
            setattr(self, gate + "_weights", np.random.randn(hidden_size, combined_size) * 0.1)
            setattr(self, gate + "_bias", np.zeros((hidden_size, 1)))

        # prediction layer
        self.prediction_weights = np.random.randn(1, hidden_size) * 0.1
        self.prediction_bias = np.zeros((1, 1))

    # activation function
    def sig(self, input_value):
        input_value = np.clip(input_value, -50, 50)
        return 1 / (1 + np.exp(-input_value))

    def gate_value(self, gate, combined_input, use_tanh=False):
        gate_weights = getattr(self, gate + "_weights")
        gate_bias = getattr(self, gate + "_bias")

        value = np.dot(gate_weights, combined_input) + gate_bias

        if use_tanh:
            return np.tanh(value)

        return self.sig(value)

    def sequence(self, input_sequence):
        hidden_state = np.zeros((self.hidden_size, 1))
        memory_state = np.zeros((self.hidden_size, 1))

        for price_value in input_sequence:
            current_price = np.array([[price_value]])
            combined_input = np.vstack((hidden_state, current_price))

            forget_gate = self.gate_value("forget", combined_input)
            input_gate = self.gate_value("input", combined_input)

            new_memory = self.gate_value("memory", combined_input, use_tanh=True)

            memory_state = forget_gate * memory_state + input_gate * new_memory

            output_gate = self.gate_value("output", combined_input)
            hidden_state = output_gate * np.tanh(memory_state)

        predicted_value = np.dot(self.prediction_weights, hidden_state) + self.prediction_bias

        return predicted_value, hidden_state


    def train(self, training_inputs, training_targets, total_loops=20):
        error_history = []

        for current_epoch in range(total_loops):
            total_error = 0

            for row_index in range(len(training_inputs)):
                predicted_value, hidden_state = self.sequence(training_inputs[row_index])

                prediction_error = predicted_value - training_targets[row_index]
                total_error = total_error + (prediction_error ** 2).item()

                # update final layer
                prediction_weight_gradient = 2 * prediction_error * hidden_state.T
                prediction_bias_gradient = 2 * prediction_error

                self.prediction_weights = self.prediction_weights - self.learning_rate * prediction_weight_gradient
                self.prediction_bias = self.prediction_bias - self.learning_rate * prediction_bias_gradient

            epoch_RMSE = np.sqrt(total_error / len(training_inputs))
            error_history.append(epoch_RMSE)

            print("Epoch", current_epoch + 1, "scaled train RMSE:", round(epoch_RMSE, 6))

        return error_history


    def predict_values(self, input_data):
        predicted_list = []

        for input_sequence in input_data:
            predicted_value, hidden_state = self.sequence(input_sequence)
            predicted_list.append(predicted_value.item())

        return np.array(predicted_list)

# model training
model = LSTM(input_size=1, hidden_size=hidden_size, learning_rate=learning_rate)

print()
print("Training model...")
training_errors = model.train(train_inputs, train_targets, total_loops=total_loops)


# model prediction
print()
print("Making predictions...")

scaled_train_predictions = model.predict_values(train_inputs)
scaled_test_predictions = model.predict_values(test_inputs)

# reverse scaling
def restored_price(scaled_values, lowest_price, highest_price):
    return scaled_values * (highest_price - lowest_price) + lowest_price

# changing data back
train_targets = restored_price(train_targets, lowest_price, highest_price)
test_targets = restored_price(test_targets, lowest_price, highest_price)

train_predictions = restored_price(scaled_train_predictions, lowest_price, highest_price)
test_predictions = restored_price(scaled_test_predictions, lowest_price, highest_price)

# error function
def RMSE(actual_values, predicted_values):
    return np.sqrt(np.mean((actual_values - predicted_values) ** 2))

# final result
train_RMSE = RMSE(train_targets, train_predictions)
test_RMSE = RMSE(test_targets, test_predictions)

print()
print("Final results")
print("----------------------")
print("Training RMSE:", round(train_RMSE, 4))
print("Testing RMSE:", round(test_RMSE, 4))

if test_RMSE > train_RMSE * 2:
    print("Note: test error is much bigger than train error")

if test_RMSE < train_RMSE:
    print("Note: test error is lower than train error for this stock period")


# save prediction result
prediction_output = pd.DataFrame({
    "Actual Close": test_targets.flatten(),
    "Predicted Close": test_predictions.flatten()
})

prediction_output.to_csv("predictions.csv", index=False)


# save experiment
experiment_log = pd.DataFrame({
    "Exp Num": [1],
    "Ticker": [stock_name],
    "Start Date": [start_date],
    "Window": [window_size],
    "Hidden": [hidden_size],
    "Rate": [learning_rate],
    "Loops": [total_loops],
    "Train Size": [len(train_inputs)],
    "Test Size": [len(test_inputs)],
    "Train RMSE": [train_RMSE],
    "Test RMSE": [test_RMSE],
    "Note": ["Simplified LSTM, only final layer trained"]
})

experiment_log.to_csv("experiment_log.csv", index=False)


# final plot
plt.figure(figsize=(12, 6))
plt.plot(test_targets, label="Original Price")
plt.plot(test_predictions, label="Predicted Price")
plt.title("Predicted Vs Original")
plt.xlabel("Test Example")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.savefig("predicted_vs_actual.png")
plt.close()


# training error plot
plt.figure(figsize=(12, 6))
plt.plot(training_errors)
plt.title("Training Error")
plt.xlabel("Epoch")
plt.ylabel("Scaled RMSE")
plt.tight_layout()
plt.savefig("training_error.png")
plt.close()


print()
print("Saved files")
print("predictions.csv")
print("experiment_log.csv")
print("predicted_vs_actual.png")
print("training_error.png")
