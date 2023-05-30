import warnings
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import requests
from io import BytesIO
from streamlit_option_menu import option_menu
from streamlit import session_state as state

# Initialize history list if it doesn't exist
if "history" not in state:
    state.history = []

# Set the page configuration to wide layout
st.set_page_config(layout="wide")

# Load the preprocessed fashion data
clothes_data = pd.read_pickle("preprocessed_fashion_data_16k")

# Load the word2vec model
with open('word2vec_model', 'rb') as handle:
    model = pickle.load(handle)

# Initialize the title vectorizers
bow_title_vectorizer = CountVectorizer()
bow_title_features = bow_title_vectorizer.fit_transform(clothes_data['title'])

tfidf_title_vectorizer = TfidfVectorizer(min_df=0)
tfidf_title_features = tfidf_title_vectorizer.fit_transform(clothes_data['title'])

idf_title_vectorizer = CountVectorizer()
idf_title_features = idf_title_vectorizer.fit_transform(clothes_data['title'])

# Initialize the brand and color vectorizers
clothes_data['brand'].fillna(value="Not Given", inplace=True)

brands = [x.replace(" ", "-") for x in clothes_data['brand'].values]
colors = [x.replace(" ", "-") for x in clothes_data['color'].values]

brand_vectorizer = CountVectorizer()
brand_features = brand_vectorizer.fit_transform(brands)

color_vectorizer = CountVectorizer()
color_features = color_vectorizer.fit_transform(colors)

#getting all the keys from w2v model
vocab = model.keys()

# making a word2vec function

def build_avg_vec(sentence, num_features, doc_id, m_name):

    featureVec = np.zeros((num_features), dtype="float32")
    nwords = 0
   
    for word in sentence.split():
        nwords += 1
        if word in vocab:
            if m_name == 'idf_weighted' and word in idf_title_vectorizer.vocabulary_:
                featureVec = np.add(featureVec, idf_title_features[doc_id, idf_title_vectorizer.vocabulary_[word]] * model[word])
            elif m_name == 'avg':
                featureVec = np.add(featureVec, model[word])
            elif m_name == 'tf_idf_weighted'and word in tfidf_title_vectorizer.vocabulary_:
                featureVec = np.add(featureVec, tfidf_title_features[doc_id, tfidf_title_vectorizer.vocabulary_[word]] * model[word])
    if(nwords>0):
        featureVec = np.divide(featureVec, nwords)
    #returns the avg vector of given sentance, its of shape (1, 300)
    return featureVec

# an average word2vec

doc_id = 0
avg_w2v_title = []
#for every title we build a avg vector representation
for i in clothes_data['title']:
    avg_w2v_title.append(build_avg_vec(i, 300, doc_id,'avg'))
    doc_id += 1

#avg_w2v_title = np.array(# number of doc in courpus * 300), each row corresponds to a doc
avg_w2v_title = np.array(avg_w2v_title)

# IDF weighted word2vec

doc_id = 0
idf_w2v_title_weight = []
#for every title we build a weighted vector representation
for i in clothes_data['title']:
    idf_w2v_title_weight.append(build_avg_vec(i, 300, doc_id,'idf_weighted'))
    doc_id += 1
#w2v_title = np.array(# number of doc in courpus * 300), each row corresponds to a doc
idf_w2v_title_weight = np.array(idf_w2v_title_weight)

# TF IDF weighted word2vec

doc_id = 0
tf_idf_w2v_title_weight = []
#for every title we build a weighted vector representation
for i in clothes_data['title']:
    tf_idf_w2v_title_weight.append(build_avg_vec(i, 300, doc_id,'tf_idf_weighted'))
    doc_id += 1
#w2v_title = np.array(# number of doc in courpus * 300), each row corresponds to a doc 
tf_idf_w2v_title_weight = np.array(tf_idf_w2v_title_weight)

# Load the bottleneck features and ASINS
bottleneck_features_train = np.load('IMAGE_features_CNN.npy')
asins = np.load('ASINS_features_for_16k.npy')

asins = list(asins)
df_asins = list(clothes_data['asin'])

def get_similar_products(title_technique, doc_id, wt, wb, wc, wi, num_results):
    doc_id = asins.index(df_asins[doc_id])
   
    title_dist = pairwise_distances(title_technique, title_technique[doc_id].reshape(1,-1))
    brand_dist = pairwise_distances(brand_features, brand_features[doc_id])
    color_dist = pairwise_distances(color_features, color_features[doc_id])
    bottleneck_features_dist = pairwise_distances(bottleneck_features_train, bottleneck_features_train[doc_id].reshape(1,-1))
   
    pairwise_dist = (wt * title_dist +  wb * brand_dist + wc * color_dist + wi * bottleneck_features_dist) / float(wt + wb + wc + wi)

    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists = np.sort(pairwise_dist.flatten())[0:num_results]
    
    cols = st.columns(2)  # Divide the screen into two columns for side-by-side display
    
    for i in range(len(indices)):
        rows = clothes_data[['medium_image_url','title','brand','color']].loc[clothes_data['asin'] == asins[indices[i]]]
        for indx, row in rows.iterrows():
            with cols[i % 2]:  # Alternate between columns for each output
                image = Image.open(BytesIO(requests.get(row['medium_image_url']).content))
                st.image(image)
                st.write('Product Title:', row['title'])
                st.write('Brand:', row['brand'])
                st.write('Color:', row['color'])
                st.write('Amazon URL: www.amazon.com/dp/' + asins[indices[i]])
                st.write("---")  # Add a separator between products



selected = option_menu(
        menu_title = None,
        options = ["Product Recommender", "History"],
        icons = ['bag','clock-history'],
        default_index=0,
        orientation='horizontal',
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {
                "font-size": "25px",
                "text-align": "left",
                "margin":"0px",
                "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "green"},
        }
)
# Handle menu selection
if selected == "Product Recommender":
    # Define the main heading
    st.title('AMAZON FASHION RECOMMENDER')
    st.markdown(':+1:')

    # create sidebar inputs
    titles = clothes_data['title'].tolist()
    titles.insert(0,'')
    search_text = st.sidebar.selectbox('Select a Title', titles)

    if search_text !="":
        state.history.append(search_text)

    # Define the available title techniques and their corresponding weights
    title_techniques = {
        'Bag of Words (BoW)': {
            'title_vectorizer': bow_title_vectorizer,
            'title_features': bow_title_features,
            'weights': np.array([60, 100, 10, 30])
        },
        'TF IDF': {
            'title_vectorizer': tfidf_title_vectorizer,
            'title_features': tfidf_title_features,
            'weights': np.array([1, 100, 10, 80])
        },
        'Average Word2Vec': {
            'title_vectorizer': None,
            'title_features': avg_w2v_title,
            'weights': np.array([100, 10, 10, 80])  # Adjust the weights as needed
        },
        'IDF Weighted Word2Vec': {
            'title_vectorizer': None,
            'title_features': idf_w2v_title_weight,
            'weights': np.array([10, 50, 10, 80])  # Adjust the weights as needed
        },
        'TF IDF Weighted Word2Vec': {
            'title_vectorizer': None,
            'title_features': tf_idf_w2v_title_weight,
            'weights': np.array([1, 50, 100, 10])
        }
    }

    # Retrieve the selected title technique
    selected_title_technique = st.sidebar.selectbox("Select Title Technique", list(title_techniques.keys()))

    # Retrieve the weights for the selected title technique
    weights = title_techniques[selected_title_technique]['weights']

    # Perform similarity analysis when the user clicks the "Get Recommendations" button
    if selected_title_technique:
    # Check if a title is selected
        if search_text:
            # Find the index of the selected title
            doc_id = np.where(clothes_data['title'] == search_text)[0]
        
            # Check if the selected title exists in the dataset
            if len(doc_id) > 0:
                # Get the selected title technique's vector representation
                title_features = title_techniques[selected_title_technique]['title_features']
        
                # Call the get_similar_products function with the selected title technique and weights
                get_similar_products(title_features, doc_id[0], weights[0], weights[1], weights[2], weights[3], num_results=10)
        else:
            st.write("Select a product.")
    else:
        st.write("Please select a title.")
    pass

if selected == "History":
    # Display the history of titles
    st.title("HISTORY")
    if st.button("Clear History"):
        state.history = []
        st.write("History cleared.")
        # Clear the saved titles
        #st.session_state["titles"] = []
    st.write("Recent Titles:")
    history_style = '''
        <style>
            .history-title-container {
                width: 300px;
                text-align: center;
            }
            .history-title {
                margin: 5px 0;
                text-align: left;
            }
        </style>
    '''
    st.markdown(history_style, unsafe_allow_html=True)
    st.markdown('<div class="history-title-container">', unsafe_allow_html=True)
    for title in reversed(state.history):
        st.markdown(f'<div class="history-title">{title}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
  