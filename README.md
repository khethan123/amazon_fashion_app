# Amazon Fashion App

Clothes recommending streamlit app

This project basically uses 5 types of methods to just see similar images. <br>
Here's a brief explanation of each of the five methods used:

1. **Bag of Words (BoW) Based Product Similarity**: BoW is a pre-processing technique that can generate a numeric form from an input text. It converts text into fixed-length vectors by counting how many times each word appears. The resulting vectors can be used to calculate the similarity between two documents. However, BoW does not bring in any information on the meaning of the text.

2. **TF IDF Based Product Similarity**: Term Frequency Inverse Document Frequency (TFIDF) works by proportionally increasing the number of times a word appears in the document but is counterbalanced by the number of documents in which it is present. Hence, words like 'this', 'are' etc., that are commonly present in all the documents are not given a very high rank. However, a word that is present too many times in a few of the documents will be given a higher rank as it might be indicative of the context of the document. It looks at how many times a word appears in a document and how many documents it appears in. If a word appears many times in a document but not in many other documents, it is probably an important word for that document. If a word appears many times in all documents, it is probably not very important. It’s like finding out which words are the most special and unique to each document.

3. **IDF Based Product Similarity**: IDF measures the log of total documents present in the text corpus divided by the number of documents that contain the particular word. It is used to weigh down the frequent terms while scaling up the rare ones. The idea behind IDF is that words that appear in many documents are not very important, while words that appear in few documents are more important. 

4. **Average Word2Vec Based Product Similarity**: Word2Vec is a neural network-based technique that learns vector representations of words from large corpora. Average Word2Vec calculates the average vector representation of all words in a sentence or document and uses this to calculate similarity between two documents. To represent a document as a vector, we take the average of the vectors of all the words in the document. This gives us a single vector that represents the document.

The resulting vector can be used to calculate similarity between two documents using cosine similarity. Documents that are similar will have vectors that are closer together in the high-dimensional space, while documents that are dissimilar will have vectors that are farther apart.

Average Word2Vec is useful when we want to compare documents based on their *meaning*, rather than just their frequency of occurrence of words. It is widely used in natural language processing applications such as text classification, sentiment analysis, and recommendation systems.

5. **IDF Weighted Word2Vec Based Product Similarity**: IDF Weighted Word2Vec is similar to Average Word2Vec, but it also takes into account how important each word is to a document based on its IDF score.

These techniques are used to calculate similarity between two or more documents or products based on their textual content. They are widely used in recommendation systems, search engines, and other applications where similarity between textual data needs to be calculated.

You will get to know of the comparison once you run amazon-fashion-recommendation.ipynb, Amazon_fashions.ipynb and lastly app.py  in an order. <br>
Make sure to open any editor on your local system. Then install miniconda and create a new virtual environment and then install all the required dependencies and then install jupyter in your editor and run it. 
