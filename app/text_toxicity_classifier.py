import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score, mean_absolute_error, mean_squared_error

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Toxic Comment Classifier | Streamlit")
url = ("https://raw.githubusercontent.com/jmkho/NLP_LA01/main/app/train.csv")

@st.experimental_singleton
def dataframe_load(): 
  return pd.read_csv(url)

# Cleaning the dataset 
alphanum = lambda x: re.sub("\w*\d\w*", " ", x)
punc = lambda x: re.sub("[%s]" % re.escape(string.punctuation), " ", x.lower())
enter = lambda x: re.sub("\n", " ", x)
non_ascii = lambda x: re.sub(r"[^\x00-\x7f]", r" ", x)
dataframe_load()['comment_text'] = dataframe_load()['comment_text'].map(alphanum).map(punc).map(enter).map(non_ascii)

# Separating dataset comments 
data_tox = dataframe_load().loc[:,['id','comment_text','toxic']]
data_sev = dataframe_load().loc[:,['id','comment_text','severe_toxic']]
data_obs = dataframe_load().loc[:,['id','comment_text','obscene']]
data_thr = dataframe_load().loc[:,['id','comment_text','threat']]
data_ins = dataframe_load().loc[:,['id','comment_text','insult']]
data_iht = dataframe_load().loc[:,['id','comment_text','identity_hate']]

data_tox_1 = data_tox[data_tox['toxic'] == 1].iloc[0:5000, :]
data_tox_0 = data_tox[data_tox['toxic'] == 0].iloc[0:5000, :]
data_tox_done = pd.concat([data_tox_1, data_tox_0], axis = 0)

data_sev_1 = data_sev[data_sev['severe_toxic'] == 1].iloc[0:1595,:]
data_sev_0 = data_sev[data_sev['severe_toxic'] == 0].iloc[0:1595,:]
data_sev_done = pd.concat([data_sev_1, data_sev_0], axis = 0)

data_obs_1 = data_obs[data_obs['obscene'] == 1].iloc[0:5000,:]
data_obs_0 = data_obs[data_obs['obscene'] == 0].iloc[0:5000,:]
data_obs_done = pd.concat([data_obs_1, data_obs_0], axis=0)

data_ins_1 = data_ins[data_ins['insult'] == 1].iloc[0:5000,:]
data_ins_0 = data_ins[data_ins['insult'] == 0].iloc[0:5000,:]
data_ins_done = pd.concat([data_ins_1, data_ins_0], axis=0)

data_thr_1 = data_thr[data_thr['threat'] == 1].iloc[0:478,:]
data_thr_0 = data_thr[data_thr['threat'] == 0].iloc[0:1912,:]  
data_thr_done = pd.concat([data_thr_1, data_thr_0], axis=0)

data_iht_1 = data_iht[data_iht['identity_hate'] == 1].iloc[0:1405,:]
data_iht_0 = data_iht[data_iht['identity_hate'] == 0].iloc[0:5620,:]
data_iht_done = pd.concat([data_iht_1, data_iht_0], axis=0)



# streamlit properties
model_tox = st.container()
model_sev = st.container()
model_obs = st.container()
model_ins = st.container()
model_thr = st.container()
model_iht = st.container()

with model_tox:
  x_tox = data_tox_done.comment_text
  y_tox = data_tox_done['toxic']

  x_train_tox, x_test_tox, y_train_tox, y_test_tox = train_test_split(x_tox, y_tox, test_size = 0.3, random_state=42)

  vec1 = TfidfVectorizer(ngram_range = (1,1), stop_words='english')
  x_train_fit_tox = vec1.fit_transform(x_train_tox)
  x_test_fit_tox = vec1.transform(x_test_tox)

  SVC1 = LinearSVC(random_state=0, tol=1e-05)
  CLF1 = CalibratedClassifierCV(SVC1) 
  CLF1.fit(x_train_fit_tox, y_train_tox)
  y_pred_tox = CLF1.predict(x_test_fit_tox)


with model_sev:
  x_sev = data_sev_done.comment_text
  y_sev = data_sev_done['severe_toxic']

  x_train_sev, x_test_sev, y_train_sev, y_test_sev = train_test_split(x_sev, y_sev, test_size = 0.3, random_state=42)

  vec2 = TfidfVectorizer(ngram_range =(1,1), stop_words='english')
  x_train_fit_sev = vec2.fit_transform(x_train_sev)
  x_test_fit_sev = vec2.transform(x_test_sev)

  SVC2 = LinearSVC(random_state=0, tol=1e-05)
  CLF2 = CalibratedClassifierCV(SVC2) 
  CLF2.fit(x_train_fit_sev, y_train_sev)
  y_pred_sev = CLF2.predict(x_test_fit_sev)


with model_obs:
  x_obs = data_obs_done.comment_text
  y_obs = data_obs_done['obscene']

  x_train_obs, x_test_obs, y_train_obs, y_test_obs = train_test_split(x_obs, y_obs, test_size = 0.3, random_state=42)

  vec3 = TfidfVectorizer(ngram_range =(1,1), stop_words='english')

  x_train_fit_obs = vec3.fit_transform(x_train_obs)
  x_test_fit_obs = vec3.transform(x_test_obs)

  SVC3 = LinearSVC(random_state=0, tol=1e-05)
  CLF3 = CalibratedClassifierCV(SVC3) 
  CLF3.fit(x_train_fit_obs, y_train_obs)
  y_pred_obs = CLF3.predict(x_test_fit_obs)


with model_ins:
  x_ins = data_ins_done.comment_text
  y_ins = data_ins_done['insult']

  x_train_ins, x_test_ins, y_train_ins, y_test_ins = train_test_split(x_ins, y_ins, test_size = 0.3, random_state=42)

  vec4 = TfidfVectorizer(ngram_range =(1,1), stop_words='english')

  x_train_fit_ins = vec4.fit_transform(x_train_ins)
  x_test_fit_ins = vec4.transform(x_test_ins)

  SVC4 = LinearSVC(random_state=0, tol=1e-05)
  CLF4 = CalibratedClassifierCV(SVC4) 
  CLF4.fit(x_train_fit_ins, y_train_ins)
  y_pred_ins = CLF4.predict(x_test_fit_ins)


with model_thr:
  x_thr = data_thr_done.comment_text
  y_thr = data_thr_done['threat']

  x_train_thr, x_test_thr, y_train_thr, y_test_thr = train_test_split(x_thr, y_thr, test_size = 0.3, random_state=42)

  vec5 = TfidfVectorizer(ngram_range =(1,1), stop_words='english')

  x_train_fit_thr = vec5.fit_transform(x_train_thr)
  x_test_fit_thr = vec5.transform(x_test_thr)

  SVC5 = LinearSVC(random_state=0, tol=1e-05)
  CLF5 = CalibratedClassifierCV(SVC5) 
  CLF5.fit(x_train_fit_thr, y_train_thr)
  y_pred_thr = CLF5.predict(x_test_fit_thr)


with model_iht:
  x_iht = data_iht_done.comment_text
  y_iht = data_iht_done['identity_hate']

  x_train_iht, x_test_iht, y_train_iht, y_test_iht = train_test_split(x_iht, y_iht, test_size = 0.3, random_state=42)

  vec6 = TfidfVectorizer(ngram_range =(1,1), stop_words='english')

  x_train_fit_iht = vec6.fit_transform(x_train_iht)
  x_test_fit_iht = vec6.transform(x_test_iht)

  SVC6 = LinearSVC(random_state=0, tol=1e-05)
  CLF6 = CalibratedClassifierCV(SVC6) 
  CLF6.fit(x_train_fit_iht, y_train_iht)
  y_pred_iht = CLF6.predict(x_test_fit_iht)

@st.experimental_memo
def f_acc(a, b):
  acc = accuracy_score(a, b)*100
  return acc

@st.experimental_memo
def f_hamm(a, b):
  hamm = hamming_loss(a, b)
  return hamm

@st.experimental_memo
def f_jacc(a, b):
  jacc = jaccard_score(a, b)
  return jacc

@st.experimental_memo
def f_mae(a, b):
  mae = mean_absolute_error(a, b)
  return mae

@st.experimental_memo
def f_rmse(a, b):
  rmse = float(format(np.sqrt(mean_squared_error(a, b)), '.2f'))
  return rmse

acc_tox = f_acc(y_test_tox, y_pred_tox)
acc_sev = f_acc(y_test_sev, y_pred_sev)
acc_obs = f_acc(y_test_obs, y_pred_obs)
acc_ins = f_acc(y_test_ins, y_pred_ins)
acc_thr = f_acc(y_test_thr, y_pred_thr)
acc_iht = f_acc(y_test_iht, y_pred_iht)

hamm_tox = f_hamm(y_test_tox, y_pred_tox)
hamm_sev = f_hamm(y_test_sev, y_pred_sev)
hamm_obs = f_hamm(y_test_obs, y_pred_obs)
hamm_ins = f_hamm(y_test_ins, y_pred_ins)
hamm_thr = f_hamm(y_test_thr, y_pred_thr)
hamm_iht = f_hamm(y_test_iht, y_pred_iht)

jacc_tox = f_jacc(y_test_tox, y_pred_tox)
jacc_sev = f_jacc(y_test_sev, y_pred_sev)
jacc_obs = f_jacc(y_test_obs, y_pred_obs)
jacc_ins = f_jacc(y_test_ins, y_pred_ins)
jacc_thr = f_jacc(y_test_thr, y_pred_thr)
jacc_iht = f_jacc(y_test_iht, y_pred_iht)
  
mae_tox = f_mae(y_test_tox, y_pred_tox)
mae_sev = f_mae(y_test_sev, y_pred_sev)
mae_obs = f_mae(y_test_obs, y_pred_obs)
mae_ins = f_mae(y_test_ins, y_pred_ins)
mae_thr = f_mae(y_test_thr, y_pred_thr)
mae_iht = f_mae(y_test_iht, y_pred_iht)

rmse_tox = f_rmse(y_test_tox, y_pred_tox)
rmse_sev = f_rmse(y_test_sev, y_pred_sev)
rmse_obs = f_rmse(y_test_obs, y_pred_obs)
rmse_ins = f_rmse(y_test_ins, y_pred_ins)
rmse_thr = f_rmse(y_test_thr, y_pred_thr)
rmse_iht = f_rmse(y_test_iht, y_pred_iht)

acc_score_arr = [acc_tox, acc_sev, acc_obs, acc_ins, acc_thr, acc_iht]
hamm_score_arr = [hamm_tox, hamm_sev, hamm_obs, hamm_ins, hamm_thr, hamm_iht]
jacc_score_arr = [jacc_tox, jacc_sev, jacc_obs, jacc_ins, jacc_thr, jacc_iht]
mae_score_arr = [mae_tox, mae_sev, mae_obs, mae_ins, mae_thr, mae_iht]
rmse_score_arr = [rmse_tox, rmse_sev, rmse_obs, rmse_ins, rmse_thr, rmse_iht]

df_st = dataframe_load()

with st.sidebar:
  select = option_menu (
    menu_title = "Menu",
    options = ["Project Description", "Model Evaluation", "Try Predicting", "Clear Cache"],
  )

if select == "Project Description":
  st.title("Toxic Comment Classifier")
  st.write("The world we are living in now has advanced so much in terms of technology. Everything has become digitally accessible and everyone is enabled to access it anywhere and anytime, with little to no hassle. That creates a freedom for people to give and take information as they want, although it is not entirely positive. Rules and netiquette are running all over the internet \nbut still, it is quite likely we stumble upon negative and toxic contents.")

  st.header("Data Information")
  st.text("We use a Toxic Comment Classification Dataset provided by \nJigsaw in Kaggle. You can see head of the dataset we use below.")
  st.write(df_st.head())

  df_new = df_st.iloc[:, 2:].sum()
  plot = sns.barplot(x=df_new.index, y=df_new.values, palette="Blues_d")
  plt.title("No. of comments in each class", size=12)
  plt.xlabel("Categories", fontsize=10)
  plt.ylabel("Comments frequency", fontsize=10)

  counts = plot.patches
  labels = df_new.values

  for count, label in zip(counts, labels):
    height = count.get_height()
    plot.text(count.get_x() + count.get_width()/2, height+5, label, ha='center', va='bottom')

  st.pyplot(plt.gcf())

if select == "Model Evaluation":
  x = np.arange(6)
  st.header("Model Evaluation Score")

  plt.title("Accuracy Score of Each Category")
  plot = plt.barh(x, acc_score_arr)
  plt.xlabel("Percentage (%)", size=12)
  plt.xticks(np.arange(0, 110, 10))
  plt.yticks(x, ('Toxic', 'Severe Toxic', 'Obscene', 'Insult', 'Threat', 'Identity Hate'), size=12)
  st.pyplot(plt.gcf())

  all_score = {"Accuracy_Score (%)":acc_score_arr, "Hamming Loss":hamm_score_arr, "Jaccard Score":jacc_score_arr, "MAE Score":mae_score_arr, "R2 Score":rmse_score_arr}
  all_df = pd.DataFrame(all_score)
  all_df = all_df.rename({0:'Toxic', 1:'Severe_Toxic', 2:'Obscene', 3:'Insult', 4:'Threat', 5:'Identity_hate'})
  st.write(all_df)

  st.caption("Accuracy Score: the score of the number of correctly classified cases to the total of cases under evaluation.")
  st.caption("Hamming Loss: the fraction of targets that are misclassified.")
  st.caption("Jaccard Score: the intersection to the size of the union of label classes between predicted labels and ground truth labels.")
  st.caption("MAE Score: the difference between the measured value and true value.")
  st.caption("R2 Score: how well a regression model can predict the value of the response variable in percentage terms.")


if select == "Try Predicting":
  form = st.form(key='submit_text')
  comment = form.text_input("Enter your comment: ")
  submit = form.form_submit_button("Check toxicity")

  if submit:
    pred1 = vec1.transform([comment])
    pred2 = vec2.transform([comment])
    pred3 = vec3.transform([comment])
    pred4 = vec4.transform([comment])
    pred5 = vec5.transform([comment])
    pred6 = vec6.transform([comment])

    res_tox = CLF1.predict_proba(pred1)[:, 1]
    res_sev = CLF2.predict_proba(pred2)[:, 1]
    res_obs = CLF3.predict_proba(pred3)[:, 1]
    res_ins = CLF4.predict_proba(pred4)[:, 1]
    res_thr = CLF5.predict_proba(pred5)[:, 1]
    res_iht = CLF6.predict_proba(pred6)[:, 1]

    st.write("Toxicity percentage: %.2f %%" % (res_tox*100))
    st.write("Severe toxicity percentage: %.2f %%" % (res_sev*100))
    st.write("Obscene percentage: %.2f %%" % (res_obs*100))
    st.write("Insult percentage: %.2f %%" % (res_ins*100))
    st.write("Threat percentage: %.2f %%" % (res_thr*100))
    st.write("Identity hate percentage: %.2f %%" % (res_iht*100))



if select =="Clear Cache":
  st.text("Please help us by clearing the cache values before you leave this app")
  if st.button("Clear All"):
    st.experimental_memo.clear()
    st.experimental_singleton.clear()

    st.write("Cache cleaned. Thank you for helping us!")


