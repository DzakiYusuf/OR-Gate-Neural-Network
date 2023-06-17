import numpy as np
import tkinter as tk
import time
from tkinter import ttk

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
weights = np.array([0.506, 0.533, 0.58])

def initialize_weights():
    global weights
    weights = np.round(np.random.uniform(low=-1, high=1, size=3), 3)
    rounded_weights = np.round(weights, 3)
    updated_weights_label.config(text=f"Updated Weights: {rounded_weights}")

def restart_iteration():
    global current_iteration, iteration_counter, total_error, sequence_error, learning_rate, calculation
    iteration_counter = 1
    total_error = 0
    sequence_error = 0
    initialize_weights()
    input1.delete(0, tk.END)
    input1.insert(0, '0')
    input2.delete(0, tk.END)
    input2.insert(0, '0')
    calculate_button_pressed()
    threshold_button_pressed()
    window.update()
    iteration_counter_label.config(text=f"Epoch: {iteration_counter}")
    window.update()
    next_iteration_button.config(state=tk.DISABLED)
    update_button.config(state=tk.DISABLED)
    test_button.config(state=tk.DISABLED)
    test_output_label.config(text="Output: ")
    sequence_error_label.config(text="Total Error: 0")
    iterate_button.config(state=tk.NORMAL)
    learning_rate_label.config(text="Learning Rate: 0.1")
    learning_rate = 0.1
    calculation_button.config(state=tk.NORMAL)
    window.update()

# Set the learning rate
learning_rate = 0.1

# Create a function to update the learning rate
def update_learning_rate():
    global learning_rate
    learning_rate = float(learning_rate_entry.get())
    learning_rate_label.config(text=f"Learning Rate: {learning_rate}")

# Create the GUI
def calculate_output():
    input_values = [int(input0.get()), int(input1.get()), int(input2.get())]
    test_data = np.array(input_values)
    calculation = np.dot(test_data, weights[:3])
    rounded_calculation = round(calculation, 3)
    calculation_label.config(text=f"Calculation: {rounded_calculation}")
    
def threshold_output():
    global current_iteration, sequence_error, error
    input_values = [int(input0.get()), int(input1.get()), int(input2.get())]
    test_data = np.array(input_values)
    calculation = np.dot(test_data, weights)
    output = activation_function(calculation)
    threshold_label.config(text=f"Threshold Output: {output}")
    
    if output == y[current_iteration]:
        if sequence_error != 0 or [X[current_iteration, 0], X[current_iteration, 1]] != [1, 1]:
            if test_output_label.cget('text') != "FINISH":
                test_output_label.config(text="Output: TRUE")
                update_button.config(state=tk.DISABLED)
                next_iteration_button.config(state=tk.NORMAL)
                test_button.config(state=tk.DISABLED)
              
            else:
                test_output_label.config(text="FINISH")
                update_button.config(state=tk.DISABLED)
                next_iteration_button.config(state=tk.NORMAL)
                test_button.config(state=tk.NORMAL)
                iterate_button.config(state=tk.DISABLED)
        else:
            test_output_label.config(text="FINISH")
            update_button.config(state=tk.DISABLED)
            next_iteration_button.config(state=tk.NORMAL)
            test_button.config(state=tk.NORMAL)
            iterate_button.config(state=tk.DISABLED)
            calculation_button.config(state=tk.DISABLED)
    else:
        test_output_label.config(text="Error! Press update weight")
        next_iteration_button.config(state=tk.DISABLED)
        update_button.config(state=tk.NORMAL)
        sequence_error += 1
        sequence_error_label.config(text=f"Total Error: {sequence_error}")
        
        iterate_button.config(state=tk.NORMAL)
    
    threshold_button.config(state=tk.DISABLED)

def threshold_button_pressed():
    threshold_output()
    
def update_weights():
    global weights, current_iteration, total_error, output, iteration_counter, sequence_error, state
    input_values = [int(input0.get()), int(input1.get()), int(input2.get())]
    test_data = np.array(input_values)
    calculation = np.dot(test_data, weights[:3])
    output = activation_function(calculation)

    error = np.abs(y[current_iteration] - output)
    total_error += error

    input_values = [X[current_iteration, 0], X[current_iteration, 1]]
    
    input_values = [X[current_iteration, 0], X[current_iteration, 1]]
    if input_values == [0, 1]:
       
        weights[0] = weights[0] + (learning_rate * (y[1] - output) * 1)
        weights[2] = weights[2] + (learning_rate * (y[1] - output) * 1)
    elif input_values == [1, 0]:
        
        weights[0] = weights[0] + (learning_rate * (y[2] - output) * 1)
        weights[1] = weights[1] + (learning_rate * (y[2] - output) * 1)
    elif input_values == [0, 0]:
      
        weights[0] = weights[0] + (learning_rate * (y[0] - output) * 1)
    elif input_values == [1, 1]:
      
        weights[0] = weights[0] + (learning_rate * (y[3] - output) * 1)
        weights[1] = weights[1] + (learning_rate * (y[3] - output) * 1)
        weights[2] = weights[2] + (learning_rate * (y[3] - output) * 1)

    rounded_weights = np.round(weights, 3)
    updated_weights_label.config(text=f"Updated Weights: {rounded_weights}")

    
    total_error += error

    if input_values == [1, 1]:
        sequence_error = 0

    if iteration_counter == len(X):
        if total_error == 0:
            test_output_label.config(text="Finish")
            
        total_error = 0
        next_iteration_button.config(state=tk.NORMAL)
        update_button.config(state=tk.DISABLED)  # Disable the Update Weights button
    else:
        next_iteration_button.config(state=tk.NORMAL)
        update_button.config(state=tk.NORMAL)
    update_button.config(state=tk.DISABLED)
      
def next_iteration():
    global current_iteration, iteration_counter, sequence_error, state
    input1.delete(0, tk.END)
    input2.delete(0, tk.END)
    current_iteration = (current_iteration + 1) % len(X)
    input1.insert(0, X[current_iteration, 0])
    input2.insert(0, X[current_iteration, 1])
    next_iteration_button.config(state=tk.DISABLED)
    update_button.config(state=tk.DISABLED)
    calculation_button.config(state=tk.NORMAL)

    if [X[current_iteration, 0], X[current_iteration, 1]] == [0, 0]:
        
        if sequence_error != 0:
            iteration_counter += 1
            window.update()
            iteration_counter_label.config(text=f"Epoch: {iteration_counter}")
         
            window.update()
            sequence_error = 0
            sequence_error_label.config(text="Total Error: 0")
        else:
            iteration_counter_label.config(text=f"Epoch: {iteration_counter}")
            
            window.update()
            sequence_error = 0
            sequence_error_label.config(text="Total Error: 0")
      
def test():
    input_values = [int(input0.get()), int(input1.get()), int(input2.get())]
    test_data = np.array(input_values)
    calculation = np.dot(test_data, weights[:3])
    rounded_calculation = round(calculation, 3)
    calculation_label.config(text=f"Calculation: {rounded_calculation}")
    
    test_data = np.array(input_values)
    calculation = np.dot(test_data, weights[:3])
    output = activation_function(calculation)
    threshold_label.config(text=f"Threshold Output: {output}")
    next_iteration_button.config(state=tk.NORMAL)
    

        
def calculate_button_pressed():
    calculate_output()
    threshold_button.config(state=tk.NORMAL)  # Enable the threshold button

# Define the iteration counter
iteration_counter = 1

def perform_iteration():
    global iteration_counter
    calculate_button_pressed()
    window.update()
  
    if threshold_button['state'] == tk.NORMAL:
        threshold_output()
        
        window.update()
        time.sleep(0.1)
    
    if update_button['state'] == tk.NORMAL:
        update_weights()
        window.update()
        
    if next_iteration_button['state'] == tk.NORMAL:
        next_iteration()
        time.sleep(0.1)
    window.update()

def iterate_button_pressed(num_iterations):

    # Perform the desired actions for the specified number of iterations
    
    for i in range(num_iterations):
    # while iteration_counter != num_iterations:
      if test_output_label.cget('text') == "FINISH":
        break
      perform_iteration()
      perform_iteration()
      perform_iteration()
      perform_iteration()
      window.update()
      time.sleep(0.2)  # Add a delay for better visualization
    iterate_button.config(state=tk.NORMAL)
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
input1 = tk.Entry(window, state='normal')
input1.grid(row=1, column=1)
input1.insert(0, X[0, 0])

input2_label = tk.Label(window, text="X2:")
input2_label.grid(row=2, column=0)
input2 = tk.Entry(window)
input2.grid(row=2, column=1)
input2.insert(0, X[0, 0])

calculation_button = tk.Button(window, text="Calculate", command=calculate_button_pressed)
calculation_button.grid(row=3, column=0, pady=5)

calculation_label = tk.Label(window, text="Calculation: ")
calculation_label.grid(row=3, column=1)

threshold_button = tk.Button(window, text="Threshold", command=threshold_output)
threshold_button.grid(row=4, column=0, pady=5)
threshold_button.config(state=tk.DISABLED)

threshold_label = tk.Label(window, text="Threshold Output: ")
threshold_label.grid(row=4, column=1)

update_button = tk.Button(window, text="Update Weights", command=update_weights)
update_button.grid(row=6, column=0, pady=5)
update_button.config(state=tk.DISABLED)

updated_weights_label = tk.Label(window, text="Weights: [0.506, 0.533, 0.58]\n")
updated_weights_label.grid(row=6, column=1, sticky="w")


iteration_counter = 1
iteration_counter_label = tk.Label(window, text="Epoch: 1")
iteration_counter_label.grid(row=8, column=1)

next_iteration_button = tk.Button(window, text="Next Input", command=next_iteration)
next_iteration_button.grid(row=7, column=0, pady=5)
next_iteration_button.config(state=tk.DISABLED)

test_button = tk.Button(window, text="Test", command=test)
test_button.grid(row=1, column=2, pady=5)
test_button.config(state=tk.DISABLED)

test_output_label = tk.Label(window, text="Output: ")
test_output_label.grid(row=5, column=0)

sequence_error = 0
sequence_error_label = tk.Label(window, text="Total Error: 0")
sequence_error_label.grid(row=7, column=1)

restart_button = tk.Button(window, text="Restart", command=restart_iteration)
restart_button.grid(row=12, column=1, pady=5)

learning_rate_label = tk.Label(window, text=f"Learning Rate: {learning_rate}")
learning_rate_label.grid(row=9, column=0, sticky="w")

learning_rate_entry = tk.Entry(window, width=5)
learning_rate_entry.grid(row=10, column=0)
learning_rate_entry.insert(0, "0.1")  # Add a placeholder text "0.1"

learning_rate_button = tk.Button(window, text="Update Miu", command=update_learning_rate)
learning_rate_button.grid(row=11, column=0)
current_iteration = 0
total_error = 0

# Create a label and entry field for the number of iterations
iterations_label = tk.Label(window, text="Auto iteration:")
iterations_label.grid(row=9, column=1, padx=10, pady=5)

iterations_entry = tk.Entry(window, width=5)
iterations_entry.grid(row=9, column=2, padx=10, pady=5)
iterations_entry.insert(0, "10")

# Create the iterate button
iterate_button = tk.Button(window, text="Iterate", command=lambda: iterate_button_pressed(int(iterations_entry.get())))
iterate_button.grid(row=9, column=3, pady=5, sticky="W")
window.mainloop()
