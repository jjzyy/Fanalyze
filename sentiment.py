import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import praw

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Setting global variable of num records to make it easier to run
num_records = 25

nba_teams_formatted = {
    'bostonceltics': 'Boston Celtics', 'GoNets': 'Brooklyn Nets', 'NYKnicks': 'New York Knicks', 'sixers': 'Philadelphia 76ers', 'torontoraptors': 'Toronto Raptors', 'chicagobulls': 'Chicago Bulls', 'clevelandcavs': 'Cleveland Cavaliers', 'DetroitPistons': 'Detroit Pistons', 'pacers': 'Indiana Pacers', 'MkeBucks': 'Milwaukee Bucks', 'AtlantaHawks': 'Atlanta Hawks', 'CharlotteHornets': 'Charlotte Hornets', 'heat': 'Miami Heat', 'OrlandoMagic': 'Orlando Magic', 'washingtonwizards': 'Washington Wizards', 'denvernuggets': 'Denver Nuggets', 'timberwolves': 'Minnesota Timberwolves', 'Thunder': 'Oklahoma City Thunder', 'ripcity': 'Portland Trailblazers', 'UtahJazz': 'Utah Jazz', 'Warriors': 'Golden State Warriors', 'LAClippers': 'Los Angeles Clippers', 'Lakers': 'Los Angeles Lakers', 'suns': 'Phoenix Suns', 'kings': 'Sacramento Kings', 'mavericks': 'Dallas Mavericks', 'Rockets': 'Houston Rockets', 'memphisgrizzlies': 'Memphis Grizzlies', 'NOLAPelicans': 'New Orleans Pelicans', 'NBASpurs': 'San Antonio Spurs'
}

reddit = praw.Reddit(
    client_id='Ubr_Qomo-ICNsZJmJh5icQ',
    client_secret='MTXK1xjKQOQDoPfBSWSlarECPSeBcQ',
    user_agent='jzyNBAS'
)

nba_teams = ['bostonceltics', 'GoNets', 'NYKnicks', 'sixers', 'torontoraptors', 'chicagobulls', 'clevelandcavs', 'DetroitPistons', 'pacers', 'MkeBucks', 'AtlantaHawks', 'CharlotteHornets', 'heat', 'OrlandoMagic', 'washingtonwizards', 'denvernuggets', 'timberwolves', 'Thunder', 'ripcity', 'UtahJazz', 'Warriors', 'LAClippers', 'Lakers', 'suns', 'kings', 'mavericks', 'Rockets', 'memphisgrizzlies', 'NOLAPelicans', 'NBASpurs']

@st.cache_data
def fetch_and_analyze_sentiments():
    sentiment_data = []

    for team_name in nba_teams:
        subreddit_name = team_name
        subreddit = reddit.subreddit(subreddit_name)

        # Retrieve and store the top/hot/new submissions from each NBA team subreddit
        for submission in subreddit.hot(limit=num_records):
            if submission.selftext:  # Check if there is any text in the submission
                # Analyze sentiment for each submission body
                scores = analyzer.polarity_scores(submission.selftext)
                sentiment_data.append({
                    'team_name': subreddit_name,
                    'body': submission.selftext,
                    'sentiment_scores': scores,
                    'compound_score': scores['compound']  # Store only the compound score
                })

    # Create a DataFrame from the sentiment data
    sentiment_df = pd.DataFrame(sentiment_data)

    # Group by 'team_name' and calculate the mean of compound scores for the first 50 records of each team
    mean_grouped_df = sentiment_df.groupby('team_name')['compound_score'].apply(lambda x: x.head(num_records).mean())
    std_grouped_df = sentiment_df.groupby('team_name')['compound_score'].apply(lambda x: x.head(num_records).std())
    var_grouped_df = sentiment_df.groupby('team_name')['compound_score'].apply(lambda x: x.head(num_records).var())

    # Create a new DataFrame with the mean compound scores
    normalized_scored_df = pd.DataFrame({
        'team_name': mean_grouped_df.index,
        'mean_compound_score': mean_grouped_df.values * 100,
        'std_compound_score': std_grouped_df.values * 100,
        'var_compound_score': var_grouped_df.values * 100,
    })

    return normalized_scored_df

# Fetch and analyze sentiment data
normalized_scored_df = fetch_and_analyze_sentiments()

# Streamlit app layout
st.title("NBA Team Sentiment Analysis")

# Display the dataframe
st.write("Sentiment Analysis Data")
st.dataframe(normalized_scored_df)

# Plot of mean compound scores by NBA team
st.write("Mean Compound Sentiment Score by NBA Team")
sns.set_palette("deep")
normalized_scored_df_sorted = normalized_scored_df.sort_values(by='mean_compound_score', ascending=False)

plt.figure(figsize=(11, 6))
sns.barplot(data=normalized_scored_df_sorted, x='team_name', y='mean_compound_score')
plt.title('Mean Compound Sentiment Score by NBA Team')
plt.xlabel('NBA Team')
plt.ylabel('Mean Compound Sentiment Score')
plt.xticks(rotation=90)
plt.tight_layout()  # Adjust layout to prevent overlap of labels
st.pyplot(plt.gcf())  # Display the plot in Streamlit

# Plot of mean vs. standard deviation of sentiment scores by NBA team
st.write("Mean vs. Standard Deviation of Sentiment Scores by NBA Team")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=normalized_scored_df, x='mean_compound_score', y='std_compound_score', hue='team_name', legend=False)
plt.title('Mean vs. Standard Deviation of Sentiment Scores by NBA Team')
plt.xlabel('Mean Compound Sentiment Score')
plt.ylabel('Standard Deviation of Compound Sentiment Score')
plt.tight_layout()
st.pyplot(plt.gcf())  # Display the plot in Streamlit
