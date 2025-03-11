# Review-sentimental-analysis
# Emotion and Sentiment Analysis Using Transformers

## Overview
This project utilizes the `transformers` library to perform emotion classification on text using a pre-trained model. It takes user input, analyzes emotions for each sentence, and visualizes the results using bar graphs. Additionally, it scrapes product reviews from a given website and determines whether a product is recommended based on the sentiment analysis of the reviews.

## Features
- Uses `bhadresh-savani/distilbert-base-uncased-emotion` model for emotion classification.
- Splits input text into sentences for detailed analysis.
- Displays emotions with confidence scores.
- Generates bar charts for each sentence to visualize emotions.
- Scrapes product reviews from a given URL.
- Aggregates emotion scores across multiple reviews to provide a recommendation.

## Prerequisites
Ensure you have the following dependencies installed before running the script:

```sh
pip install transformers torch matplotlib regex requests beautifulsoup4
```

## Usage
1. Run the script:
   ```sh
   python script.py
   ```
2. Enter the product review webpage URL when prompted.
3. The script scrapes reviews and analyzes their emotions.
4. View the overall emotion classification results and bar charts for each review.
5. The script provides a final recommendation on whether the product is worth buying based on the sentiment scores.

## Code Explanation
- Loads the pre-trained emotion classification model from `transformers`.
- Scrapes product reviews from a given URL using `BeautifulSoup`.
- Splits each review into sentences and analyzes their emotions.
- Aggregates emotion scores across multiple reviews.
- Compares positive and negative emotions to determine the final recommendation.
- Visualizes emotion distribution using `matplotlib` bar charts.

## Example Output
```
Enter the website URL: https://example.com/reviews

Analyzing product reviews...

Text: This product is amazing!
Emotion: joy, Score: 0.92
...
(Bar chart displayed)

Text: It broke after two days. Very disappointed.
Emotion: sadness, Score: 0.85
...
(Bar chart displayed)

Overall Emotion Scores:
Joy: 3.45
Sadness: 2.78
...

✅ Recommendation: This product is recommended based on the reviews.
```

## License
This project is open-source and available for modification and distribution.
