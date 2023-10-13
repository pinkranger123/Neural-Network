import matplotlib.pyplot as plt

# Define your data
hidden_neurons = [3, 4, 5]
learning_rates = [0.01, 0.001, 0.0001]
accuracy_5_fold = [accuracy_3_0_01, accuracy_3_0_001, accuracy_3_0_0001,
                   accuracy_4_0_01, accuracy_4_0_001, accuracy_4_0_0001,
                   accuracy_5_0_01, accuracy_5_0_001, accuracy_5_0_0001]
accuracy_10_fold = [accuracy_3_0_01_10, accuracy_3_0_001_10, accuracy_3_0_0001_10,
                    accuracy_4_0_01_10, accuracy_4_0_001_10, accuracy_4_0_0001_10,
                    accuracy_5_0_01_10, accuracy_5_0_001_10, accuracy_5_0_0001_10]

x = range(len(hidden_neurons) * len(learning_rates))

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

# Bar chart for 5-fold cross-validation
ax1.bar(x, accuracy_5_fold, color='b', width=0.4, label='5-Fold Cross Validation')
ax1.set_xticks(x)
ax1.set_xticklabels(['3_0.01', '3_0.001', '3_0.0001', '4_0.01', '4_0.001', '4_0.0001', '5_0.01', '5_0.001', '5_0.0001'])
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Accuracy for 5-Fold Cross Validation')
ax1.legend()

# Bar chart for 10-fold cross-validation
ax2.bar(x, accuracy_10_fold, color='g', width=0.4, label='10-Fold Cross Validation')
ax2.set_xticks(x)
ax2.set_xticklabels(['3_0.01', '3_0.001', '3_0.0001', '4_0.01', '4_0.001', '4_0.0001', '5_0.01', '5_0.001', '5_0.0001'])
ax2.set_xlabel('Hidden Neurons and Learning Rate')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy for 10-Fold Cross Validation')
ax2.legend()

plt.tight_layout()
plt.show()
