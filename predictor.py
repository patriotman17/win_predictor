import pandas as pd

# Function to reset team records for each new season
def reset_team_records_for_new_season(team_records):
    for team in team_records.keys():
        team_records[team] = {'wins': 0, 'losses': 0, 'games': []}

# Function to determine if a team is 'good' or 'bad' based on their record
def is_team_good(team_record):
    return team_record['wins'] > team_record['losses']

# Adjusted function to update records and determine the quality of both home and visitor teams
def update_records_and_get_team_quality_corrected(row, team_records, season_tracker):
    home_team = row['home_display_name']
    visitor_team = row['visitor_display_name']
    home_score = row['home_score_point_total']
    visitor_score = row['visitor_score_point_total']
    season = row['season']

    # Reset records if we are in a new season
    if season_tracker['last_season'] != season:
        reset_team_records_for_new_season(team_records)
        season_tracker['last_season'] = season

    # Initialize the record for teams if not already present
    if home_team not in team_records:
        team_records[home_team] = {'wins': 0, 'losses': 0, 'games': []}
    if visitor_team not in team_records:
        team_records[visitor_team] = {'wins': 0, 'losses': 0, 'games': []}

    # Determine if teams are 'good' or 'bad' based on their record before this game
    home_is_good = is_team_good(team_records[home_team])
    visitor_is_good = is_team_good(team_records[visitor_team])

    # Update the records based on the game result
    if home_score > visitor_score:
        team_records[home_team]['wins'] += 1
        team_records[visitor_team]['losses'] += 1
    elif home_score < visitor_score:
        team_records[home_team]['losses'] += 1
        team_records[visitor_team]['wins'] += 1

    # Add the game to the list of games that these teams have played
    team_records[home_team]['games'].append({'date': row['game_date'], 'opponent': visitor_team, 'home_game': True})
    team_records[visitor_team]['games'].append({'date': row['game_date'], 'opponent': home_team, 'home_game': False})

    return pd.Series([home_is_good, visitor_is_good])

# Load the dataset
file_path = 'game_boxscores_2016_2023.csv'
boxscores_df = pd.read_csv(file_path)

# Selecting the relevant columns including team names
selected_columns = [
    'season', 'season_type', 'week', 'game_date', 
    'home_team_id', 'visitor_team_id', 
    'home_score_point_total', 'visitor_score_point_total', 
    'home_timeouts_remaining', 'visitor_timeouts_remaining', 
    'validated', 'home_display_name', 'visitor_display_name'
]

# Creating a new dataframe with only the selected columns
selected_boxscores_df = boxscores_df[selected_columns]

# Initialize a dictionary to hold the team records and a dictionary to track the last processed season
team_records = {}
season_tracker = {'last_season': None}

# Sort the dataframe by season and then by game_date to ensure chronological order
sorted_boxscores_df = selected_boxscores_df.sort_values(by=['season', 'game_date'])

# Apply the function to each row in the dataframe
sorted_boxscores_df[['home_is_good', 'visitor_is_good']] = sorted_boxscores_df.apply(
    lambda row: update_records_and_get_team_quality_corrected(row, team_records, season_tracker), axis=1
)

# Display the first few rows of the dataframe with the updated columns
sorted_boxscores_df[['game_date', 'season', 'home_display_name', 'visitor_display_name', 'home_score_point_total', 'visitor_score_point_total', 'home_is_good', 'visitor_is_good']].head()


# %% Exploratory Analysis

# Step 1: Summary statistics for the numerical columns in the dataset
numerical_summary = sorted_boxscores_df.describe()

# Step 2: Visualizing the distribution of total points scored by home and away teams
import matplotlib.pyplot as plt

# Histogram of home team total points
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.hist(sorted_boxscores_df['home_score_point_total'].dropna(), bins=30, alpha=0.7, color='blue')
plt.title('Distribution of Home Team Points')
plt.xlabel('Points')
plt.ylabel('Number of Games')

# Histogram of visitor team total points
plt.subplot(1, 2, 2)
plt.hist(sorted_boxscores_df['visitor_score_point_total'].dropna(), bins=30, alpha=0.7, color='red')
plt.title('Distribution of Visitor Team Points')
plt.xlabel('Points')
plt.ylabel('Number of Games')

plt.tight_layout()
plt.show()

numerical_summary, plt

import seaborn as sns

# Selecting numerical columns for the correlation matrix
numerical_columns = [
    'home_score_point_total', 'visitor_score_point_total',
    'home_timeouts_remaining', 'visitor_timeouts_remaining'
]

# Calculating the correlation matrix
corr_matrix = sorted_boxscores_df[numerical_columns].corr()

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Correlation between home and visitor scores
score_corr = sorted_boxscores_df[['home_score_point_total', 'visitor_score_point_total']].corr()

# Relationship between 'good' teams and their scoring
sorted_boxscores_df['home_good_and_score'] = sorted_boxscores_df['home_is_good'].astype(int) * sorted_boxscores_df['home_score_point_total']
sorted_boxscores_df['visitor_good_and_score'] = sorted_boxscores_df['visitor_is_good'].astype(int) * sorted_boxscores_df['visitor_score_point_total']

good_team_score_corr = sorted_boxscores_df[['home_good_and_score', 'visitor_good_and_score']].corr()

# Impact of being a 'good' team on winning
# We will calculate win percentages for 'good' home and visitor teams
# We need to create a column indicating who won the game first
sorted_boxscores_df['home_win'] = sorted_boxscores_df['home_score_point_total'] > sorted_boxscores_df['visitor_score_point_total']
sorted_boxscores_df['visitor_win'] = sorted_boxscores_df['visitor_score_point_total'] > sorted_boxscores_df['home_score_point_total']

# Now we calculate win percentages
home_good_win_percentage = sorted_boxscores_df[sorted_boxscores_df['home_is_good']]['home_win'].mean()
visitor_good_win_percentage = sorted_boxscores_df[sorted_boxscores_df['visitor_is_good']]['visitor_win'].mean()

# Explore home field advantage
home_win_percentage = sorted_boxscores_df['home_win'].mean()

score_corr, good_team_score_corr, home_good_win_percentage, visitor_good_win_percentage, home_win_percentage

# Visualization of win percentages for 'good' home and visitor teams
win_percentages = pd.Series({
    'Home Good Team Win %': home_good_win_percentage,
    'Visitor Good Team Win %': visitor_good_win_percentage,
    'Overall Home Team Win %': home_win_percentage
})

# Bar plot for win percentages
win_percentages.plot(kind='bar', color=['blue', 'red', 'green'])
plt.title('Win Percentages')
plt.ylabel('Win Percentage')
plt.xlabel('Team and Good Status')
plt.xticks(rotation=45)
plt.show()

# Improved visualization for 'good' status vs points scored for home and visitor teams
plt.figure(figsize=(14, 7))

# Boxplot for home teams
plt.subplot(1, 2, 1)
sns.boxplot(data=sorted_boxscores_df, x='home_is_good', y='home_score_point_total')
plt.title('Home Team Good Status vs Score')
plt.xlabel('Is Home Team Good (1 = Yes, 0 = No)')
plt.ylabel('Home Team Score')

# Boxplot for visitor teams
plt.subplot(1, 2, 2)
sns.boxplot(data=sorted_boxscores_df, x='visitor_is_good', y='visitor_score_point_total')
plt.title('Visitor Team Good Status vs Score')
plt.xlabel('Is Visitor Team Good (1 = Yes, 0 = No)')
plt.ylabel('Visitor Team Score')

plt.tight_layout()
plt.show()

# %% Creating the Test Model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# For the sake of simplicity, we'll only use 'home_is_good' and 'visitor_is_good' as features for our initial model
# We also need to ensure that we're only using regular season data to predict the playoffs
regular_season_data = sorted_boxscores_df[sorted_boxscores_df['season_type'] == 'REG']

# Our features will include whether the home and visitor teams are 'good' according to their regular season performance
X = regular_season_data[['home_is_good', 'visitor_is_good']]

# The target variable will be whether the home team won
y = regular_season_data['home_win']

# Since we don't have the 2023 data yet, we'll split the current data into training and "pseudo-testing" sets
# The real test will occur when we have the 2023 data
X_train, X_pseudo_test, y_train, y_pseudo_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_pseudo_test_scaled = scaler.transform(X_pseudo_test)

# Initialize and train the logistic regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train_scaled, y_train)

# Make predictions (this will be replaced with the actual test set predictions when we have the 2023 data)
y_pred = logreg_model.predict(X_pseudo_test_scaled)

# Evaluate the model using pseudo-test set
accuracy = accuracy_score(y_pseudo_test, y_pred)
classification_rep = classification_report(y_pseudo_test, y_pred)

accuracy, classification_rep

# %% Adding the test 2023 dataset

# Function to determine if a team is 'good' based on their record
def is_team_good(team_record):
    total_games = team_record['wins'] + team_record['losses'] + team_record['ties']
    if total_games == 0:
        return False  # Can't determine 'good' status if no games played
    win_percentage = (team_record['wins'] + 0.5 * team_record['ties']) / total_games
    return win_percentage > 0.5

# Function to update team records game by game for the 2023 season
def update_team_records_2023(boxscores_df, team_records):
    # Create a copy of the dataframe to avoid modifying the original data
    updated_boxscores_df = boxscores_df.copy()
    
    # Iterate through each game and update the team records
    for index, row in updated_boxscores_df.iterrows():
        home_team = row['home_display_name']
        visitor_team = row['visitor_display_name']
        home_score = row['home_score_point_total']
        visitor_score = row['visitor_score_point_total']

        # Initialize team records if they don't exist
        if home_team not in team_records:
            team_records[home_team] = {'wins': 0, 'losses': 0, 'ties': 0}
        if visitor_team not in team_records:
            team_records[visitor_team] = {'wins': 0, 'losses': 0, 'ties': 0}

        # Update the 'good' status based on records before this game
        home_is_good = is_team_good(team_records[home_team])
        visitor_is_good = is_team_good(team_records[visitor_team])
        updated_boxscores_df.at[index, 'home_is_good'] = home_is_good
        updated_boxscores_df.at[index, 'visitor_is_good'] = visitor_is_good

        # Update records based on the game result
        if home_score > visitor_score:
            team_records[home_team]['wins'] += 1
            team_records[visitor_team]['losses'] += 1
        elif home_score < visitor_score:
            team_records[home_team]['losses'] += 1
            team_records[visitor_team]['wins'] += 1
        else:  # Handle potential ties
            team_records[home_team]['ties'] += 1
            team_records[visitor_team]['ties'] += 1

    return updated_boxscores_df

# Load the dataset for the 2023 season
file_path_2023 = 'game_boxscores_2023_1.csv'
boxscores_2023_df = pd.read_csv(file_path_2023)

# Convert game_date to datetime and sort by game_date and season
boxscores_2023_df['game_date'] = pd.to_datetime(boxscores_2023_df['game_date'])
boxscores_2023_df_sorted = boxscores_2023_df.sort_values(by=['game_date', 'season'])

# Reset the team records for the start of the 2023 season
team_records_2023 = {team: {'wins': 0, 'losses': 0, 'ties': 0} for team in team_records}

# Update the team records for the 2023 season
updated_boxscores_2023_df = update_team_records_2023(boxscores_2023_df_sorted, team_records_2023)

# %% Making our predictions

# Making predictions for the 2023 season using the logistic regression model
# We'll use the updated 'home_is_good' and 'visitor_is_good' features to make the predictions

# Scale the 2023 features using the scaler from the 2016-2022 model
X_2023 = updated_boxscores_2023_df[['home_is_good', 'visitor_is_good']]
X_2023_scaled = scaler.transform(X_2023)

# Making predictions using the trained model
y_pred_2023 = logreg_model.predict(X_2023_scaled)

# Adding the predictions to the 2023 dataframe
updated_boxscores_2023_df['predicted_home_win'] = y_pred_2023

# Displaying the first few rows with the predictions
updated_boxscores_2023_df[['game_date', 'week', 'season', 'home_display_name', 'visitor_display_name', 'home_is_good', 'visitor_is_good', 'predicted_home_win']].head(10)

# %% Checking our predictions

# Function to determine the actual result of the game
def get_actual_result(row):
    if row['home_score_point_total'] > row['visitor_score_point_total']:
        return True  # Home team won
    else:
        return False  # Home team lost or game tied

# Apply the function to get the actual result for each game
updated_boxscores_2023_df['actual_home_win'] = updated_boxscores_2023_df.apply(get_actual_result, axis=1)

# Compare predictions with actual results
comparison_columns = ['game_date', 'week', 'season', 'home_display_name', 'visitor_display_name',
                      'home_score_point_total', 'visitor_score_point_total',
                      'predicted_home_win', 'actual_home_win']
comparison_df = updated_boxscores_2023_df[comparison_columns]

# Calculate the number of correct predictions
correct_predictions_count = comparison_df[comparison_df['predicted_home_win'] == comparison_df['actual_home_win']].shape[0]

# Calculate the total number of games
total_games = comparison_df.shape[0]

# Calculate the accuracy of the model
accuracy = correct_predictions_count / total_games

# Display the comparison dataframe, the number of correct predictions, and the accuracy
print(f"Model Accuracy for 2023 Weeks 1-10: {accuracy:.2%}")
comparison_df.head(10), correct_predictions_count

