from collections import Counter
import matplotlib.pyplot as plt
import polars as pl

df = pl.read_csv("TIBKAT_dataset/core_train.csv")

data = df["subjects"].str.split(by=" ").explode()
print(data)
subjects = []
frequency = []
# Count occurrences
counter = Counter(data)
for key, count in counter.items():
    subjects.append(key)
    frequency.append(count)
new_df = pl.DataFrame({"subject":subjects,"frequency":frequency})
new_df.write_csv("train_label_dist.csv")


# Get the most common 20 labels and group the rest into "Others"
# most_common = counter.most_common(20)
# other_count = sum(count for label, count in counter.items() if label not in dict(most_common))

# # Prepare data for visualization
# labels, counts = zip(*most_common)
# labels = list(labels) + ["Others"]
# counts = list(counts) + [other_count]

# # plt.figure(figsize=(12, 6))
# # plt.bar(labels, counts, color="skyblue")
# # plt.xticks(rotation=45, ha="right")
# # plt.xlabel("Labels")
# # plt.ylabel("Frequency")
# # plt.title("Top 20 Label Distribution with 'Others'")
# # plt.tight_layout()
# # plt.show()
# # plt.savefig("img.jpg")
# plt.figure(figsize=(10, 6))
# plt.hist(counter.values(), bins=50, color="lightcoral", edgecolor="black")
# plt.xlabel("Frequency")
# plt.ylabel("Number of Labels")
# plt.title("Distribution of Label Frequencies")
# plt.savefig("img.jpg")