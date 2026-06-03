import nltk
import string
import json
import traceback
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) 
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# 2. Load FAQs from the external JSON file
try:
    with open('faqs.json', 'r', encoding='utf-8') as file:
        faqs = json.load(file)
except FileNotFoundError:
    print("⚠️ ERROR: 'faqs.json' not found! Make sure it is in the same folder as this script.")
    faqs = {"Error": "The FAQ database is missing."}

faq_questions = list(faqs.keys())
faq_answers = list(faqs.values())

# Initialize the NLP Lemmatizer
lemmer = nltk.stem.WordNetLemmatizer()

# 3. Text Preprocessing Function
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def Normalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# 4. Intent Matching via Cosine Similarity
def get_bot_response(user_input, history):
    try:
        faq_questions.append(user_input)
        
        TfidfVec = TfidfVectorizer(tokenizer=Normalize, stop_words='english', token_pattern=None)
        tfidf = TfidfVec.fit_transform(faq_questions)
        
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx = vals.argsort()[0][-2]
        
        flat = vals.flatten()
        flat.sort()
        highest_similarity_score = flat[-2]
        
        faq_questions.pop()
        
        # If the score is very low, it means the bot couldn't find a match
        if highest_similarity_score == 0:
            return "I am sorry! I don't quite understand what you are asking. Could you rephrase?"
        else:
            return faq_answers[idx]
            
    except Exception as e:
        if len(faq_questions) > len(faqs):
            faq_questions.pop()
        return f"⚠️ SYSTEM ERROR: {str(e)}\n\nCheck your terminal window for the full traceback."

# 5. Create the Chat UI using Gradio
print(f"✅ Successfully loaded {len(faqs)} FAQs.")
print("🚀 Starting Chatbot UI...")

demo = gr.ChatInterface(
    fn=get_bot_response,
    title="Bookstore FAQ Assistant",
    description="Ask me anything about shipping, returns, and payments!"
)

if __name__ == "__main__":
    demo.launch()