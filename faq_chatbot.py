import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open('faqs.json') as f: faqs=json.load(f)
questions=[q['q'] for q in faqs]; answers=[q['a'] for q in faqs]
vec=TfidfVectorizer().fit(questions); q_tfidf=vec.transform(questions)
print('FAQ Chatbot (exit to quit)')
while True:
    user=input('You: ')
    if user.lower() in ('exit','quit'): break
    sims=cosine_similarity(vec.transform([user]), q_tfidf)[0]
    best=sims.argmax()
    print('Bot:', answers[best] if sims[best]>0.2 else "I don't know.")
