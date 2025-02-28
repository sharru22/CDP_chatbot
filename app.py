from flask import Flask, render_template, request
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample documentation data
documents = {
    "segment": "To set up a new source in Segment, go to your workspace, click 'Add Source', choose the source type, and follow the setup instructions.",
    "mparticle": "To create a user profile in mParticle, navigate to the dashboard, click 'Profiles', then 'Create New Profile'. Fill in user attributes and save.",
    "lytics": "To build an audience segment in Lytics, go to the 'Audiences' tab, click 'Create New', define rules based on user behavior, and save the segment.",
    "zeotap": "To integrate your data with Zeotap, access the API dashboard, generate an API key, configure the integration settings, and connect your data sources."
}

# Convert documentation to TF-IDF format
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents.values())

# Function to get the best answer with similarity threshold
def get_best_answer(user_query):
    query_vector = vectorizer.transform([user_query])  # Convert query to TF-IDF
    similarities = cosine_similarity(query_vector, doc_vectors)  # Compare with documentation

    best_match_index = similarities.argmax()  # Find best match
    best_score = similarities.max()  # Get similarity score
    best_doc_key = list(documents.keys())[best_match_index]  # Get matching CDP

    # If similarity score is below 0.3, return "I don't know"
    if best_score < 0.3:  
        return "Sorry, I don’t have an answer for that."

    return documents[best_doc_key]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_answer", methods=["POST"])
def get_answer():
    user_question = request.form["user_question"]
    response = get_best_answer(user_question)
    return response

if __name__ == "__main__":
    app.run(debug=True)
