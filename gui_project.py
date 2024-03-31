import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from wordcloud import WordCloud
import seaborn as sns


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
def show_information_dataset():


  vote_positive = data_full[data_full["Rating"] > 7.8].shape[0] + data_tmp[data_tmp['Sentiment'] == 'Positive'].shape[0]
  vote_negative = data_full[data_full["Rating"] < 6.8].shape[0] + data_tmp[data_tmp['Sentiment'] == 'Negative'].shape[0]
  new_data_positive = pd.concat([data_full[data_full["Rating"] > 7.8], data_tmp[data_tmp['Sentiment'] == 'Positive']], axis=0, ignore_index=True)
  new_data_negative = pd.concat([data_full[data_full["Rating"] < 6.8], data_tmp[data_tmp['Sentiment'] == 'Negative']], axis=0, ignore_index=True)

  new_data_positive['Tier']='Positive'
  new_data_negative['Tier']='Negative' 
  new_data = pd.concat([new_data_positive,new_data_negative],axis=0, ignore_index=True)

  n_restaurant = new_data['IDRestaurant'].nunique()
  st.write(f'Total Vote: {new_data.shape[0]}')
  st.write(f'Total Restaurant: {n_restaurant}')
  st.write(f'Vote Positive: {vote_positive}')
  st.write(f'Vote Negative: {vote_negative}')
  st.write(f'Average Rating: {round(data_full["Rating"].mean(), 1)}')

  fig = plt.figure(figsize=(10, 6))
  ax = sns.countplot(data=data_full, x='District')
  plt.title('Distribution of Restaurants by District')
  plt.xlabel('District')
  plt.ylabel('Count')
  st.pyplot(fig)

  review_bins_num = new_data['Tier'].value_counts()
  st.dataframe(review_bins_num)
  review_bins_num = review_bins_num.reset_index()
  st.dataframe(review_bins_num)

  fig1 = plt.figure(figsize=(10, 6))
  ax = sns.barplot(data=review_bins_num, x='index', y='Tier')
  plt.title('Number of reviews according to each group of interest level')
  plt.xlabel('Tier')
  plt.ylabel('Number')
  st.pyplot(fig1)

  review_bins = new_data['Tier'].value_counts(normalize=True)*100
  fig2 = plt.figure(figsize = (15,8))
  plt.pie(x=review_bins, labels=review_bins.index, autopct='%1.1f%%')
  plt.legend(title = "Group:")
  plt.title('Number of reviews according to each group')
  st.pyplot(fig2)
    
  
  if vote_positive > 0:
    st.write('WordCloud Positive:')
    wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=30).generate(' '.join(new_data_positive['Comment'].tolist()))
    st.image(wordcloud.to_array(), width=600)
  if vote_negative > 0:
    st.write('WordCloud Negative:')
    wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=30).generate(' '.join(new_data_negative['Comment'].tolist()))
    st.image(wordcloud.to_array(), width=600)                               
def show_information_restaurant(id):
  try:
    data_check = data_full[data_full['IDRestaurant'] == id]
    st.write(f'Name Restaurant: {data_check.iloc[0]["Restaurant"]}')
    st.write(f'Address Restaurant: {data_check.iloc[0]["Address"]}')
    st.write(f'Time: : {data_check.iloc[0]["Time_x"]}')
    st.write(f'Price: : {data_check.iloc[0]["Price"]}')
    st.write(f'District: : {data_check.iloc[0]["District"]}')
    st.write(f'Total Vote: {data_check.shape[0]}')

    vote_positive = data_check[data_check["Rating"] > 7.8].shape[0] + data_tmp[((data_tmp['Sentiment'] == 'Positive') & (data_tmp['IDRestaurant'] == id))].shape[0]
    vote_negative = data_check[data_check["Rating"] < 6.8].shape[0] + data_tmp[((data_tmp['Sentiment'] == 'Negative') & (data_tmp['IDRestaurant'] == id))].shape[0]
    st.write(f'Vote Positive: {vote_positive}')
    st.write(f'Vote Negative: {vote_negative}')
    st.write(f'Average Rating: {round(data_check["Rating"].mean(), 1)}')
    new_data_positive = pd.concat([data_check[data_check["Rating"] > 7.8], data_tmp[((data_tmp['Sentiment'] == 'Positive') & (data_tmp['IDRestaurant'] == id))]], axis=0, ignore_index=True)
    new_data_negative = pd.concat([data_check[data_check["Rating"] < 6.8], data_tmp[((data_tmp['Sentiment'] == 'Negative') & (data_tmp['IDRestaurant'] == id))]], axis=0, ignore_index=True)
    
    new_data_positive['Tier']='Positive'
    new_data_negative['Tier']='Negative' 
    new_data = pd.concat([new_data_positive,new_data_negative],axis=0, ignore_index=True)

    fig = plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=data_full, x='District')
    plt.title('Distribution of Restaurants by District')
    plt.xlabel('District')
    plt.ylabel('Count')
    st.pyplot(fig)

    review_bins_num = new_data['Tier'].value_counts()
    review_bins_num = review_bins_num.reset_index()

    fig1 = plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=review_bins_num, x=review_bins_num.index, y='Tier')
    plt.title('Number of reviews according to each group of interest level')
    plt.xlabel('Tier')
    plt.ylabel('Number')
    st.pyplot(fig1)

    review_bins = new_data['Tier'].value_counts(normalize=True)*100
    fig2 = plt.figure(figsize = (15,8))
    plt.pie(x=review_bins, labels=review_bins.index, autopct='%1.1f%%')
    plt.legend(title = "Group:")
    plt.title('Number of reviews according to each group')
    st.pyplot(fig2)

    if vote_positive > 0:
      st.write('WordCloud Positive:')
      wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=30).generate(' '.join(new_data_positive['Comment'].tolist()))
      st.image(wordcloud.to_array(), width=600)
    if vote_negative > 0:
      st.write('WordCloud Negative:')
      wordcloud = WordCloud(background_color='white', stopwords=stopwords, max_words=30).generate(' '.join(new_data_negative['Comment'].tolist()))
      st.image(wordcloud.to_array(), width=600)
  except:
    st.write(f'ID Restaurant: {id} not found')
# Write code create sidebar selectbox
st.set_page_config(page_title="Sentiment Analysis Application", page_icon="🧊", layout="wide", initial_sidebar_state="expanded")
st.title("Sentiment Analysis")
st.sidebar.title("Options in application Sentiment Analysis")

selectbox = st.sidebar.selectbox("Projects", ["Home","Overview Dataset", "Visualization Dataset", "Predict New Feedback", "Show Evaluation"])
if selectbox == "Home":
  st.title("Trung Tâm Tin Học")
  st.write("## Capstone Project - Đồ án tốt nghiệp Data Science")

  st.header('Requirement')
  st.write("""
  ##### Xây dựng hệ thống Sentiment Analysis.
  """)
  st.write("""Sử dụng Machine Learning, Xây dựng hệ thống hỗ trợ nhà hàng/quán
 ăn phân loại các phản hồi của khách hàng """)

  st.write("Nhóm thực hiện:")
  st.write("- Phạm Lê Phú")
  st.write("- Phạm Nguyễn Đức Huy")
elif selectbox == "Overview Dataset":
  st.header('Overview Dataset')
  show_information_dataset()
elif selectbox == "Visualization Dataset":
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
  radio_option = st.radio("Choose your option:", ["Enter your feedback", "Upload file"])
  if radio_option == "Enter your feedback":
    text_predict = st.text_area(label="Feeback", placeholder="Enter New Feedback")
    if (text_predict != "") or (text_predict is None):
      st.snow()
      predict_feedback = predict_text(text_predict)
      st.write(f'Predict Feedback: {predict_feedback}')
  else:
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv'])
    if uploaded_file is not None:
      if Path(uploaded_file.name).suffix == '.txt':
        text_predict = uploaded_file.read().decode('utf-8')
        list_text_predict = text_predict.split('\n')
        list_sentiment = list()
        if (len(list_text_predict) == 1) and (list_text_predict[0] == ''):
          st.warning('File is empty')
        elif len(list_text_predict) > 0:
          for text in list_text_predict:
            predict_feedback = predict_text(text.strip('\n'))
            list_sentiment.append(predict_feedback)
          list_feedback = pd.DataFrame({'Feedback': list_text_predict, 'Sentiment': list_sentiment})
          st.dataframe(list_feedback, width=600)
        else:
          st.warning('File is empty')
      else:
        data_feedback_csv = pd.read_csv(uploaded_file)
        try:
          data_feedback_csv['Sentiment'] = data_feedback_csv['Feedback'].apply(lambda x: predict_text(x))
          st.dataframe(data_feedback_csv, width=600)
        except:
          st.warning('File not found or format file is not correct', icon="⚠️")
else:
  st.subheader("Show Evaluation")
  st.write("This is the classification report of Naive Bayes model (BEST model):")
  st.image('./classification_report_nb.png', width=600)
  st.write("This is the classification report of Logistic Regression model:")
  st.image('./classification_report_lr.png', width=600)
  st.write("This is the classification report of Random Forest model:")
  st.image('./classification_report_rf.png', width=600)
  st.write("This is the classification report of XGBoost model:")
  st.image('./classification_report_xgb.png', width=600)
