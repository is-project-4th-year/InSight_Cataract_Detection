import matplotlib.pyplot as plt
import numpy as np

# Mock Data representing 99.52% accuracy
# [True Negative, False Positive]
# [False Negative, True Positive]
cm = np.array([[600, 3], 
               [3, 492]]) 

# Calculate percentages for the annotations
cm_sum = np.sum(cm)
cm_perc = cm / cm_sum.astype(float) * 100

fig, ax = plt.subplots(figsize=(8, 6))

# Create the heatmap manually
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

# Labels
classes = ['Healthy', 'Cataract']
tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

# Title and Labels
ax.set_title('InSight ResNet-18 Confusion Matrix\n(Test Set N=1,098)', fontsize=14, pad=20)
ax.set_ylabel('Actual Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)

# Loop over data dimensions and create text annotations.
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()