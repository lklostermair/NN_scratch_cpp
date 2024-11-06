import matplotlib.pyplot as plt

# Path to errorLog.txt in drive
file_path = "usr/drive/errorLog.txt"

# Load error data from the file
epochs = []
errors = []

with open(file_path, "r") as f:
    for line in f:
        epoch, error = map(float, line.split())
        epochs.append(epoch)
        errors.append(error)

# Plot the training curve
plt.figure(figsize=(10, 6))
plt.plot(epochs, errors, label="Training Error")
plt.xlabel("Epoch")
plt.ylabel("Recent Average Error")
plt.title("Training Curve")
plt.legend()
plt.grid(True)
plt.show()