import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
data_full = pd.read_csv('./data_full.csv')
data_tmp = pd.read_csv('./data_tmp.csv')
stopwords = list()
with open('./vietnamese-stopwords.txt', mode='r', encoding='utf-8') as f:
  for line in f:
    stopwords.append(line.strip('\n'))
with open('./model_normal_2.pkl', 'rb') as f:
  model_normal_2 = pickle.load(f)
with open('./label_encoder_normal.pkl', 'rb') as f:
  label_encoder_normal = pickle.load(f)
with open('./tfidf_normal.pkl', 'rb') as f:
  tfidf_normal = pickle.load(f)
def predict_text(text):
  return label_encoder_normal.inverse_transform(model_normal_2.predict(tfidf_normal.transform([text])))[0]
def show_information_restaurant(id):
  try:
    data_check = data_full[data_full['IDRestaurant'] == id]
    st.write(f'Name Restaurant: {data_check.iloc[0]["Restaurant"]}')
    st.write(f'Address Restaurant: {data_check.iloc[0]["Address"]}')
    st.write(f'Total Vote: {data_check.shape[0]}')
    vote_positive = data_check[data_check["Rating"] > 7.8].shape[0] + data_tmp[((data_tmp['Sentiment'] == 'Positive') & (data_tmp['IDRestaurant'] == id))].shape[0]
    vote_negative = data_check[data_check["Rating"] < 6.8].shape[0] + data_tmp[((data_tmp['Sentiment'] == 'Negative') & (data_tmp['IDRestaurant'] == id))].shape[0]
    st.write(f'Vote Positive: {vote_positive}')
    st.write(f'Vote Negative: {vote_negative}')
    st.write(f'Average Rating: {round(data_check["Rating"].mean(), 1)}')
    if vote_positive > 0:
      st.write('WordCloud Positive:')
      wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=30).generate(' '.join(data_check[data_check["Rating"] > 7.0]['Comment'].tolist()))
      st.image(wordcloud.to_array(), width=600)
    if vote_negative > 0:
      st.write('WordCloud Negative:')
      wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=30).generate(' '.join(data_check[data_check["Rating"] <= 7.0]['Comment'].tolist()))
      st.image(wordcloud.to_array(), width=600)
  except:
    st.write(f'ID Restaurant: {id} not found')
# Write code create sidebar selectbox
st.set_page_config(page_title="Sentiment Analysis Application", page_icon="🧊", layout="wide", initial_sidebar_state="expanded")
st.title("Sentiment Analysis")
st.sidebar.title("Options in application Sentiment Analysis")
selectbox = st.sidebar.selectbox("Projects", ["Visualization Dataset", "Predict New Feedback", "Show Evaluation"])
if selectbox == "Visualization Dataset":
  st.subheader("Visualization Dataset")
  id_restaurant = st.number_input(label="ID Restaurant", placeholder="Enter ID Restaurant", step=1)
  if isinstance(id_restaurant, int):
    if id_restaurant > 0:
      show_information_restaurant(id_restaurant)
    else:
      st.warning('ID Restaurant must be greater than 0', icon="⚠️")
  else:
    st.error('ID Restaurant must be integer', icon="⚠️")
elif selectbox == "Predict New Feedback":
  st.subheader("Predict New Feedback")
  text_predict = st.text_area(label="Feeback", placeholder="Enter New Feedback")
  if (text_predict != "") or (text_predict is None):
    st.snow()
    predict_feedback = predict_text(text_predict)
    st.write(f'Predict Feedback: {predict_feedback}')
else:
  st.subheader("Show Evaluation")
  st.write("This is the classification report of Naive Bayes model (BEST model):")
  st.image('./classification_report_normal_2.png', width=600)
  st.write("This is the classification report of Logistic Regression model:")
  st.image('./classification_report_lr.png', width=600)
  st.write("This is the classification report of Random Forest model:")
  st.image('./classification_report_rf.png', width=600)
  st.write("This is the classification report of XGBoost model:")
  st.image('./classification_report_xgb.png', width=600)
