import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Create an output folder if it doesn't exist
output_dir = r"C:\Users\kaziFARZANA\Documents\THESIS\BRAIN\dataset\two\outputs"
os.makedirs(output_dir, exist_ok=True)

# Function for raw dataset (train + test)
def plot_dataset_distribution(dataset_path, title, output_file):
    categories = os.listdir(os.path.join(dataset_path, 'train'))
    train_counts = [len(os.listdir(os.path.join(dataset_path, 'train', c))) for c in categories]
    test_counts = [len(os.listdir(os.path.join(dataset_path, 'test', c))) for c in categories]

    x = range(len(categories))
    width = 0.2  # slim bars

    plt.figure(figsize=(10,6))
    plt.bar(x, train_counts, width=width, label='Train', color='skyblue')
    plt.bar([i + width for i in x], test_counts, width=width, label='Test', color='salmon')

    # Add numbers on top of bars
    for i in range(len(categories)):
        plt.text(i, train_counts[i] + 5, str(train_counts[i]), ha='center', va='bottom', fontsize=9)
        plt.text(i + width, test_counts[i] + 5, str(test_counts[i]), ha='center', va='bottom', fontsize=9)

    plt.xlabel('Categories')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.xticks([i + width/2 for i in x], categories, rotation=45)

    # Move legend below the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)

    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, output_file), dpi=300)
    plt.show()
    print(f"Chart saved as: {os.path.join(output_dir, output_file)}")


# Function for augmented training (test unchanged)
def plot_augmented_distribution(train_path, test_path, title, output_file):
    categories = os.listdir(train_path)
    train_counts = [len(os.listdir(os.path.join(train_path, c))) for c in categories]
    test_counts = [len(os.listdir(os.path.join(test_path, c))) for c in categories]

    x = range(len(categories))
    width = 0.2

    plt.figure(figsize=(10,6))
    plt.bar(x, train_counts, width=width, label='Train', color='skyblue')
    plt.bar([i + width for i in x], test_counts, width=width, label='Test', color='salmon')

    # Add numbers on top of bars
    for i in range(len(categories)):
        plt.text(i, train_counts[i] + 5, str(train_counts[i]), ha='center', va='bottom', fontsize=9)
        plt.text(i + width, test_counts[i] + 5, str(test_counts[i]), ha='center', va='bottom', fontsize=9)

    plt.xlabel('Categories')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.xticks([i + width/2 for i in x], categories, rotation=45)

    # Move legend below the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)

    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, output_file), dpi=300)
    plt.show()
    print(f"Chart saved as: {os.path.join(output_dir, output_file)}")


# Paths
raw_dataset_path = r"C:\Users\kaziFARZANA\Documents\THESIS\BRAIN\dataset\two\Brain Tumor data - ayesha\Original_data"
augmented_train_path = r"C:\Users\kaziFARZANA\Documents\THESIS\BRAIN\dataset\two\Brain Tumor data - ayesha\PREPROCESSED_TRAINN_AUGMENTED"
test_path = r"C:\Users\kaziFARZANA\Documents\THESIS\BRAIN\dataset\two\Brain Tumor data - ayesha\preprocessed_testt"

# Plot for raw dataset
plot_dataset_distribution(raw_dataset_path, 'Original Dataset Distribution (EDA)', 'original_dataset_distribution.png')

# Plot for preprocessed/augmented dataset
plot_augmented_distribution(augmented_train_path, test_path, 
                            'Preprocessed / Augmented Dataset Distribution (EDA)', 
                            'augmented_dataset_distribution.png')
