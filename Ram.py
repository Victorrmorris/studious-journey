import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the dataset containing abstracts
@st.cache
def load_data():
    return pd.read_csv('abstract.csv')  # Replace with the actual dataset path

df = load_data()

# Text analysis and clustering app
st.title('Abstract Text Analysis and Clustering')

# Sidebar to choose the number of topics
num_topics = st.sidebar.slider('Select Number of Topics', 2, 10, 5)

# Preprocess the abstracts using CountVectorizer
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['Abstract'])

# Apply LDA topic modeling
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(X)

# Display topics and their top words
st.subheader('Topics and Top Words:')
for idx, topic in enumerate(lda_model.components_):
    st.write(f"Topic {idx + 1}:")
    top_words = [vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]]
    st.write(', '.join(top_words))

# Visualize word cloud for each topic
st.subheader('Word Clouds for Topics:')
for idx, topic in enumerate(lda_model.components_):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join([vectorizer.get_feature_names()[i] for i in topic.argsort()[-50:]]))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot()

# Cluster abstracts using topic probabilities
topic_probabilities = lda_model.transform(X)
df['Cluster'] = topic_probabilities.argmax(axis=1)

# Display clustered abstracts
st.subheader('Clustered Abstracts:')
selected_cluster = st.selectbox('Select Cluster', range(num_topics))
clustered_abstracts = df[df['Cluster'] == selected_cluster]['Abstract'].tolist()
st.write(clustered_abstracts)
