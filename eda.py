from wordcloud import WordCloud
from helper_functions import load_huggingface_dataset, preprocess_dataframe_column
import pandas as pd
import matplotlib.pyplot as plt

# Function to create a word cloud from a DataFrame column
def create_wordcloud(dataframe: pd.DataFrame, column: str):
    # Join the different processed titles together.
    long_string = ','.join(list(dataframe[column].values))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_image()

    return wordcloud


def create_wordclouds_by_category(df, text_column, category_column):
    """
    Create word clouds for different categories in a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    text_column (str): The name of the column containing the text to be used for word clouds.
    category_column (str): The name of the column containing the categories.
    
    Returns:
    None (displays the word clouds)
    """
    # Check if the specified columns exist in the DataFrame
    if text_column not in df.columns or category_column not in df.columns:
        raise ValueError(f"One or both of the specified columns do not exist in the DataFrame.")
    
    # Group by the specified category column
    grouped = df.groupby(category_column)[text_column].apply(' '.join).reset_index()
    
    # Generate a word cloud for each category
    for _, row in grouped.iterrows():
        category = row[category_column]
        combined_text = row[text_column]
        
        # Create a word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
        
        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {category}')
        plt.show()

if __name__ == '__main__':
    train_df, test_df = load_huggingface_dataset("solomonk/reddit_mental_health_posts", verbose=True)
    train_df['body'] = train_df['body'].astype(str)
    train_df = preprocess_dataframe_column(train_df, 'body')
    wc = create_wordcloud(train_df, 'body')
    plt.imshow(wc, interpolation='bilinear')
    plt.show()