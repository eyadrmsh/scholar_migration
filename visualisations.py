
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def extract_edge_weights_for_boxplot(data):
    """
    Extracts edge weights classified into quantile bins for visualization.
    """
    data.year = data.year.astype(int)
    bins = [0.0, 0.40, 0,60, 0.90, 1.00]
    bin_labels = ["0-40", "40-60", "60-90", "90-100"]

    collected_data = []  # List to store extracted edge weights

    for year in range(1998, 2019, 10):
        yearly_data = data[data['year'] == year].copy()
        yearly_data['n_migrations'] = pd.to_numeric(yearly_data['n_migrations'], errors='coerce')
        quantiles = yearly_data['n_migrations'].quantile(bins)

        for i in range(len(bins) - 1):
            lower_thresh, upper_thresh = quantiles.iloc[i], quantiles.iloc[i + 1]
            if i == len(bins) - 2:
                filtered_edges = yearly_data[(yearly_data['n_migrations'] > lower_thresh) &
                                             (yearly_data['n_migrations'] <= upper_thresh)]
            else:
                filtered_edges = yearly_data[(yearly_data['n_migrations'] > lower_thresh) &
                                             (yearly_data['n_migrations'] <= upper_thresh)]

            for _, row in filtered_edges.iterrows():
                collected_data.append([year, bin_labels[i], row['n_migrations']])
    df_boxplot = pd.DataFrame(collected_data, columns=['Year', 'Quantile Bin', 'Edge Weight'])
    return df_boxplot


df_boxplot = extract_edge_weights_for_boxplot(scopus)

selected_years = [1998, 2008, 2018]
df_selected = df_boxplot[df_boxplot['Year'].isin(selected_years)]
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, year in enumerate(selected_years):
    sns.boxplot(x='Quantile Bin', y='Edge Weight', hue='Quantile Bin',
                data=df_selected[df_selected['Year'] == year],
                palette="Set2", ax=axes[i])
    axes[i].set_yscale('log')  # Log scale for better visualization
    axes[i].set_title(f'Edge Weight Distribution in {year}')
    axes[i].set_ylabel('Edge Weight' if i == 0 else '')  # Show ylabel only on first plot
    axes[i].grid(True)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Quantile Bin', loc="lower center",
           ncol=4, fontsize="small", frameon=False, bbox_to_anchor=(0.5, -0.03))

plt.tight_layout()
plt.show()