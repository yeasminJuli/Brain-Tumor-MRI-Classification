import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

output_dir = r"C:\user\juli\thesis\Dataset\outputs"
os.makedirs(output_dir, exist_ok=True)

def plot_dataset_distribution(dataset_path, title, output_file):
    categories = os.listdir(os.path.join(dataset_path, 'Train'))
    train_counts = [len(os.listdir(os.path.join(dataset_path, 'Train', c))) for c in categories]
    test_counts = [len(os.listdir(os.path.join(dataset_path, 'Test', c))) for c in categories]

    x = range(len(categories))
    width = 0.2

    plt.figure(figsize=(10,6))
    plt.bar(x, train_counts, width=width, label='Train', color='skyblue')
    plt.bar([i + width for i in x], test_counts, width=width, label='Test', color='salmon')

    for i in range(len(categories)):
        plt.text(i, train_counts[i] + 5, str(train_counts[i]), ha='center', va='bottom', fontsize=9)
        plt.text(i + width, test_counts[i] + 5, str(test_counts[i]), ha='center', va='bottom', fontsize=9)

    plt.xlabel('Categories')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.xticks([i + width/2 for i in x], categories, rotation=45)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_file), dpi=300)
    plt.show()
    print(f"Chart saved as: {os.path.join(output_dir, output_file)}")

def plot_augmented_distribution(train_path, test_path, title, output_file):
    categories = os.listdir(train_path)
    train_counts = [len(os.listdir(os.path.join(train_path, c))) for c in categories]
    test_counts = [len(os.listdir(os.path.join(test_path, c))) for c in categories]

    x = range(len(categories))
    width = 0.2

    plt.figure(figsize=(10,6))
    plt.bar(x, train_counts, width=width, label='Train', color='skyblue')
    plt.bar([i + width for i in x], test_counts, width=width, label='Test', color='salmon')

    for i in range(len(categories)):
        plt.text(i, train_counts[i] + 5, str(train_counts[i]), ha='center', va='bottom', fontsize=9)
        plt.text(i + width, test_counts[i] + 5, str(test_counts[i]), ha='center', va='bottom', fontsize=9)

    plt.xlabel('Categories')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.xticks([i + width/2 for i in x], categories, rotation=45)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_file), dpi=300)
    plt.show()
    print(f"Chart saved as: {os.path.join(output_dir, output_file)}")

# Paths
raw_dataset_path = r"C:\user\juli\thesis\Dataset\Brain Tumor MRI Dataset (G,M,P)_data"
#augmented_train_path = r"C:\user\juli\thesis\Dataset\augmented_train"
test_path = r"C:\user\juli\thesis\Dataset\preprocessed_test"

plot_dataset_distribution(raw_dataset_path, 'Original Dataset Distribution (EDA)', 'original_dataset_distribution.png')

#plot_augmented_distribution(augmented_train_path, test_path,
                            #'Augmented Dataset Distribution (EDA)',
                            #'augmented_dataset_distribution.png')