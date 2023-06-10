import numpy as np
import tkinter as tk

# Define the training data
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Define the target output
y = np.array([[0],
              [1],
              [1],
              [1]])

# Define the activation function (step function)
def activation_function(x):
    return np.where(x >= 0, 1, 0)

# Initialize the weights
weights = np.array([0.706, 0.533, 0.58])

# Set the learning rate
learning_rate = 0.1

# Create the GUI
def calculate_output():
    input_values = [int(input0.get()), int(input1.get()), int(input2.get())]
    test_data = np.array(input_values)
    calculation = np.dot(test_data, weights[:3])
    calculation_label.config(text=f"Calculation: {calculation}")
   
def threshold_output():
    global current_iteration, sequence_error
    input_values = [int(input0.get()), int(input1.get()), int(input2.get())]
    test_data = np.array(input_values)
    calculation = np.dot(test_data, weights[:3])
    output = activation_function(calculation)
    threshold_label.config(text=f"Threshold Output: {output}")

    if output == y[current_iteration]:
        finish_output_label.config(text="Output: TRUE")
        update_button.config(state=tk.DISABLED)
        next_iteration_button.config(state=tk.NORMAL)
        sequence_error = 0
    else:
        finish_output_label.config(text="Output: ERROR")
        next_iteration_button.config(state=tk.DISABLED)
        update_button.config(state=tk.NORMAL)
        sequence_error += 1
        sequence_error_label.config(text=f"Sequence Error: {sequence_error}")


def update_weights():
    global weights, current_iteration, iteration_counter, total_error
    output = activation_function(np.dot(X[current_iteration], weights[:2]) + weights[2])
    error = np.abs(y[current_iteration] - output)
    total_error += error

    input_values = [X[current_iteration, 0], X[current_iteration, 1]]

    if input_values == [0, 1]:
        weights[0] -= learning_rate
        weights[2] -= learning_rate
    elif input_values == [1, 0]:
        weights[0] -= learning_rate
        weights[1] -= learning_rate
    elif input_values == [0, 0]:
        weights[0] -= learning_rate
    elif input_values == [1, 1]:
        weights[0] -= learning_rate
        weights[1] -= learning_rate
        weights[2] -= learning_rate

    updated_weights_label.config(text=f"Updated Weights: {weights}")
    iteration_counter += 1
    iteration_counter_label.config(text=f"Iteration: {iteration_counter}")

    if iteration_counter == len(X):
        finish_button.config(state=tk.NORMAL)
        if total_error == 0:
            error_label.config(text="Error: None")
        else:
            error_label.config(text=f"Error: {total_error}")
        total_error = 0
    next_iteration_button.config(state=tk.NORMAL)

def next_iteration():
    global current_iteration
    input1.delete(0, tk.END)
    input2.delete(0, tk.END)
    current_iteration = (current_iteration + 1) % len(X)
    input1.insert(0, X[current_iteration, 0])
    input2.insert(0, X[current_iteration, 1])
    next_iteration_button.config(state=tk.DISABLED)
    update_button.config(state=tk.DISABLED)

def finish():
    input_values = [int(input0.get()), int(input1.get()), int(input2.get())]
    test_data = np.array(input_values)
    calculation = np.dot(test_data, weights[:3])
    output = activation_function(calculation)

    if output == y[current_iteration]:
        finish_output_label.config(text="Output: TRUE")
    else:
        finish_output_label.config(text="Output: ERROR")

# Create the main window
window = tk.Tk()
window.title("OR Gate Neural Network Iteration")

# Create input labels and entry fields
input0_label = tk.Label(window, text="X0:")
input0_label.grid(row=0, column=0)
input0 = tk.Entry(window, state='normal')
input0.grid(row=0, column=1)
input0.insert(0, 1)

input1_label = tk.Label(window, text="X1:")
input1_label.grid(row=1, column=0)
input1 = tk.Entry(window)
input1.grid(row=1, column=1)
input1.insert(0, X[0, 0])

input2_label = tk.Label(window, text="X2:")
input2_label.grid(row=2, column=0)
input2 = tk.Entry(window)
input2.grid(row=2, column=1)
input2.insert(0, X[0, 0])

calculation_button = tk.Button(window, text="Calculate", command=calculate_output)
calculation_button.grid(row=3, column=0, pady=5)

calculation_label = tk.Label(window, text="Calculation: ")
calculation_label.grid(row=3, column=1)

threshold_button = tk.Button(window, text="Threshold", command=threshold_output)
threshold_button.grid(row=4, column=0, pady=5)

threshold_label = tk.Label(window, text="Threshold Output: ")
threshold_label.grid(row=4, column=1)

update_button = tk.Button(window, text="Update Weights", command=update_weights)
update_button.grid(row=5, column=0, pady=5)

updated_weights_label = tk.Label(window, text="Updated Weights: ")
updated_weights_label.grid(row=5, column=1)

error_label = tk.Label(window, text="Error: ")
error_label.grid(row=6, column=1)

iteration_counter = 0
iteration_counter_label = tk.Label(window, text="Iteration: 0")
iteration_counter_label.grid(row=7, column=1)

next_iteration_button = tk.Button(window, text="Next Iteration", command=next_iteration)
next_iteration_button.grid(row=8, column=0, pady=5)
next_iteration_button.config(state=tk.DISABLED)

finish_button = tk.Button(window, text="Finish", command=finish)
finish_button.grid(row=9, column=0, pady=5)
finish_button.config(state=tk.DISABLED)

finish_output_label = tk.Label(window, text="Output: ")
finish_output_label.grid(row=9, column=1)

sequence_error = 0
sequence_error_label = tk.Label(window, text="Sequence Error: 0")
sequence_error_label.grid(row=10, column=1)

current_iteration = 0
total_error = 0

window.mainloop()
