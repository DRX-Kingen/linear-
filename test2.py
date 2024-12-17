y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0]

# Tính accuracy
correct_predictions = sum([1 for true, pred in zip(y_true, y_pred) if true == pred])
total_predictions = len(y_true)
accuracy = correct_predictions / total_predictions

print(f"Accuracy: {accuracy:.2f}")