import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import pearsonr

# Load the dataset
file_path = r'C:\Users\manue\Desktop\data_visualization\premier-player-23-24.csv'
df = pd.read_csv(file_path)

# Inspect the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Handling Missing Values
print("Missing values before handling:")
print(df.isnull().sum())

# Check data types and general information about the dataset
print("\nData types and general information:")
print(df.info())

# Impute missing values with median for numerical columns and mode for categorical columns
df.fillna({
    'Gls': df['Gls'].median(),
    'Ast': df['Ast'].median(),
    'Min': df['Min'].median(),
    'xG': df['xG'].median(),
    'xAG': df['xAG'].median(),
    # Add other columns if necessary
}, inplace=True)

print("Missing values after handling:")
print(df.isnull().sum())


# Group positions into broader categories, considering only the first position listed
def map_position(pos):
    # Split the position string by commas and strip any extra spaces
    positions = [p.strip() for p in pos.split(',')]

    # Define the position priority
    position_priority = ['GK', 'DF', 'MF', 'FW']

    # Return the first matching position based on priority
    for position in positions:
        for category in position_priority:
            if category in position:
                return category
    return 'Other'


# Apply the updated function to the 'Pos' column
df['Position_Group'] = df['Pos'].apply(map_position)

# Describing location
mean_age = df['Age'].mean()
median_age = df['Age'].median()
mode_age = df['Age'].mode()[0]  # mode() returns a Series

print(f"Mean Age: {mean_age:.2f}")
print(f"Median Age: {median_age:.2f}")
print(f"Mode Age: {mode_age:.2f}")

# Variance and Standard Deviation for Age
variance_age = df['Age'].var()
std_dev_age = df['Age'].std()

print(f"Variance of Age: {variance_age:.2f}")
print(f"Standard Deviation of Age: {std_dev_age:.2f}")

# Covariance between Age and Minutes Played
cov_age_min = df[['Age', 'Min']].cov().iloc[0, 1]

# Correlation between Age and Minutes Played
cor_age_min = df[['Age', 'Min']].corr().iloc[0, 1]

print(f"Covariance between Age and Minutes Played: {cov_age_min:.2f}")
print(f"Correlation between Age and Minutes Played: {cor_age_min:.2f}")

# Descriptive Statistics
print("Descriptive Statistics:")
print(df[['Gls', 'Ast', 'PrgP', 'Min', 'Age', 'xG']].describe())

# Top 10 Goal Scorers
top_goals = df.nlargest(10, 'Gls')
top_goals['Player_Label'] = top_goals['Player'] + ' (' + top_goals['Age'].astype(int).astype(str) + ')'
plt.figure(figsize=(12, 8))
sns.barplot(x='Gls', y='Player_Label', data=top_goals, palette='viridis')
plt.title('Top 10 Goal Scorers', fontsize=16)
plt.xlabel('Goals Scored', fontsize=14)
plt.ylabel('Player (Age)', fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.grid(axis='x')
plt.show()

# Top 10 Assist Givers
top_assists = df.nlargest(10, 'Ast')
top_assists['Player_Label'] = top_assists['Player'] + ' (' + top_assists['Age'].astype(int).astype(str) + ')'
plt.figure(figsize=(12, 8))
sns.barplot(x='Ast', y='Player_Label', data=top_assists, palette='viridis')
plt.title('Top 10 Assist Givers', fontsize=16)
plt.xlabel('Assists', fontsize=14)
plt.ylabel('Player (Age)', fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.grid(axis='x')
plt.show()

# Top 10 players with the most progressive passes
top_passes = df.nlargest(10, 'PrgP')
top_passes['Player_Label'] = top_passes['Player'] + ' (' + top_passes['Age'].astype(int).astype(str) + ')'
plt.figure(figsize=(12, 8))
sns.barplot(x='PrgP', y='Player_Label', data=top_passes, palette='viridis')
plt.title('Top 10 Players by Progressive Passes', fontsize=16)
plt.xlabel('Progressive Passes', fontsize=14)
plt.ylabel('Player (Age)', fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.grid(axis='x')
plt.show()

# Distribution of Player Ages
plt.figure(figsize=(10, 6))
min_age = df['Age'].min()  # Get the minimum age
max_age = df['Age'].max()  # Get the maximum age
bins = np.arange(min_age, max_age + 2)  # Create bins starting from the minimum age
sns.histplot(df['Age'], bins=bins, color='green')
plt.xlim(min_age, max_age + 1)  # Set x-axis limits from the minimum age to one more than the maximum age
plt.title('Distribution of Player Ages', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tick_params(axis='both', labelsize=12)  # Increase font size for tick labels
plt.show()

# Distribution of Goals (Gls)
plt.figure(figsize=(10, 6))
max_gls = df['Gls'].max()
bins = np.arange(1, max_gls + 2)
sns.histplot(df['Gls'], bins=bins, color='green')
plt.xlim(1, max_gls + 1)
plt.title('Distribution of Goals Scored (Gls)', fontsize=16)
plt.xlabel('Goals Scored', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.show()

# Scatter plot for age vs. total minutes played
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Min', data=df, color='green', s=100)
sns.regplot(x='Age', y='Min', data=df, scatter=False, color='red', line_kws={"linewidth": 2})
plt.title('Total Minutes Played vs. Age', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Total Minutes Played', fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.show()
# Hypothesis Testing for Age vs. Total Minutes Played
age = df['Age']
minutes_played = df['Min']
# Perform Pearson correlation
r_min, p_min = pearsonr(age, minutes_played)
print(f"Correlation between Age and Total Minutes Played: r = {r_min:.3f}, p-value = {p_min:.3f}")

if p_min < 0.05:
    print("Reject the null hypothesis: Age significantly affects total minutes played.")
else:
    print("Fail to reject the null hypothesis: Age does not significantly affect total minutes played.")

# Scatter plot for age vs. goals scored
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Gls', data=df, color='blue', s=100)
sns.regplot(x='Age', y='Gls', data=df, scatter=False, color='red', line_kws={"linewidth": 2})
plt.title('Goals Scored vs. Age', fontsize=16)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Goals Scored', fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.show()
# Hypothesis Testing for Age vs. Goals Scored
goals_scored = df['Gls']
# Perform Pearson correlation
r_gls, p_gls = pearsonr(age, goals_scored)
print(f"Correlation between Age and Goals Scored: r = {r_gls:.3f}, p-value = {p_gls:.3f}")

if p_gls < 0.05:
    print("Reject the null hypothesis: Age significantly affects goals scored.")
else:
    print("Fail to reject the null hypothesis: Age does not significantly affect goals scored.")

# Violin Plot for Minutes Played (Min) by Position Group
plt.figure(figsize=(10, 6))
sns.violinplot(x='Position_Group', y='Min', data=df, palette='muted')
plt.title('Distribution and Density of Minutes Played (Min) by Position Group', fontsize=16)
plt.xlabel('Position Group', fontsize=14)
plt.ylabel('Minutes Played', fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.show()

# Joint Plot for Goals vs xG (Expected Goals) with grouped positions
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Gls', y='xG', hue='Position_Group', data=df, palette={'MF': 'blue', 'FW': 'orange', 'GK': 'green', 'DF': 'red'}, s=100)
sns.regplot(x='Gls', y='xG', data=df, scatter=False, color='red', line_kws={'linewidth': 2})
plt.title('Joint Plot of Goals vs Expected Goals (xG) by Position Group', fontsize=16)
plt.xlabel('Goals Scored', fontsize=14)
plt.ylabel('Expected Goals (xG)', fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.legend(title='Position Group')
plt.show()

# Creating the bubble plot
plt.figure(figsize=(10, 8))
df_filtered = df[(df['Position_Group'] == 'MF') | (df['Position_Group'] == 'FW')]
plt.scatter(df_filtered['Gls'], df_filtered['Ast'], s=df_filtered['Min'] / 10, alpha=0.6, edgecolors="w",
            color="skyblue")
plt.xlabel('Goals (Gls)', fontsize=14)
plt.ylabel('Assists (Ast)', fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.title('Goals vs Assists with Bubble Size Representing Minutes Played (Midfielders and Forwards Only)', fontsize=16)
plt.show()

# Creating the pair plot
pair_plot_data = df[['Gls', 'Ast', 'Min', 'xG']]
sns.pairplot(pair_plot_data, diag_kind='kde', plot_kws={'alpha': 0.7, 's': 80, 'edgecolor': 'k'})
plt.suptitle('Pair Plot of Goals, Assists, Minutes Played, and Expected Goals', fontsize=16, y=1.02)
plt.show()

# Select the relevant columns for PCA
pca_columns = ['Gls', 'Ast', 'xG', 'Min', 'xAG']
pca_data = df[pca_columns].dropna()

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(pca_data)

# Add PCA results back to the dataframe
df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])

# Plot the first two principal components
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', data=df_pca, alpha=0.6, color="green")
plt.title('PCA Plot: First Two Principal Components', fontsize=16)
plt.xlabel('PCA1', fontsize=14)
plt.ylabel('PCA2', fontsize=14)
plt.show()

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df_pca['Cluster'] = kmeans.fit_predict(df_pca[['PCA1', 'PCA2']])

# Plot the clusters on the PCA components
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df_pca, palette='viridis', s=100)
plt.title('K-Means Clustering: PCA Components', fontsize=16)
plt.xlabel('PCA1', fontsize=14)
plt.ylabel('PCA2', fontsize=14)
plt.legend(title='Cluster', loc='upper right')
plt.show()

# Sorting the data to get the top 10 players by Progressive Passes (PrgP)
top_prgp_players = df.sort_values(by='PrgP', ascending=False).head(10)
