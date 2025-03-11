from transformers import pipeline
import re
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from collections import defaultdict

# Load the pre-trained emotion classification pipeline
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

# Function to analyze emotions and compute aggregate scores
def analyze_emotions(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    results = [emotion_classifier(sentence)[0] for sentence in sentences]

    aggregated_scores = defaultdict(float)

    for sentence, predictions in zip(sentences, results):
        sorted_predictions = sorted(predictions, key=lambda x: x['score'])
        emotions = [emotion['label'] for emotion in sorted_predictions]
        scores = [emotion['score'] for emotion in sorted_predictions]

        # Accumulate emotion scores
        for emotion, score in zip(emotions, scores):
            aggregated_scores[emotion] += score

        # Plot emotions for the sentence
        plt.figure(figsize=(8, 4))
        plt.bar(emotions, scores, color='skyblue')
        plt.xlabel("Emotions")
        plt.ylabel("Confidence Score")
        plt.title(f"Emotion Analysis for: {sentence}")
        plt.xticks(rotation=45)
        plt.show()

    return aggregated_scores

# Function to scrape product reviews from a website
def scrape_reviews(url):
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve the webpage.")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    reviews = soup.find_all(class_="review-text")  # Adjust this based on website structure

    return [review.get_text(strip=True) for review in reviews]

# User input for website URL
url = input("Enter the website URL: ")
reviews = scrape_reviews(url)

if reviews:
    print("\nAnalyzing product reviews...")
    total_scores = defaultdict(float)

    for review in reviews:
        review_scores = analyze_emotions(review)
        for emotion, score in review_scores.items():
            total_scores[emotion] += score

    # Final decision: Recommend or not?
    positive_emotions = total_scores.get("joy", 0) + total_scores.get("surprise", 0) + total_scores.get("love", 0)
    negative_emotions = total_scores.get("anger", 0) + total_scores.get("sadness", 0) + total_scores.get("fear", 0)

    print("\nOverall Emotion Scores:")
    for emotion, score in total_scores.items():
        print(f"{emotion.capitalize()}: {score:.4f}")

    if positive_emotions > negative_emotions:
        print("\n✅ Recommendation: This product is recommended based on the reviews.")
    else:
        print("\n❌ Recommendation: This product is NOT recommended based on the reviews.")

else:
    print("No reviews found or unable to scrape the website.")
