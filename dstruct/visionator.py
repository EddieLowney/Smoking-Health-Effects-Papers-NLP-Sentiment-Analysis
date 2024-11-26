import math

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df_list = []
for filename in os.listdir("results/ASBA"):
    df = pd.read_csv(f'results/ASBA/{filename}')
    df['file'] = os.path.splitext(os.path.basename(filename))[0]
    df_list.append(df)

df = pd.concat(df_list, ignore_index = True)
df.rename(columns = {"Unnamed: 0":'topic'}, inplace = True)
print(df)
plt.figure(dpi=500)
plt.size = (12,10)


color_labels = {"Positive":"red", "Negative":"blue", "Neutral":"grey"}
df["color"] = df["label"].map(color_labels)

sns.scatterplot(
    data=df,
    x='file',         # X-axis is the 'label' column
    y='topic',   # Y-axis is the DataFrame identifier
    size='score',      # Dot size is proportional to 'score'
    hue='label',       # Dot color represents the 'label'
    palette=color_labels,
    legend='brief',
    alpha=0.7
)

# Customize the plot
plt.title("Sentiment and Confidence Score per Category", fontsize=10)
plt.xlabel("Academic Paper", fontsize=8)
plt.ylabel("Topic", fontsize=8)
plt.xticks(rotation=90)  # Rotate X-axis labels for readability
# Adjust legend
plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()