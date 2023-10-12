import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
from keras import models
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

iris = pd.read_csv("iris.csv")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #e6dd27;  /* Koyu sarı renk */
    </style>
    """,
    unsafe_allow_html=True
)


# Sekmeleri ekleyin
selected_tab = st.sidebar.selectbox("Select Page", ["EDA", "Prediction"])

if selected_tab == "EDA":
   
    #st.sidebar.header('User Input Parameters for EDA')
    # EDA ile ilgili kodları burada ekleyin, veri görselleştirmeleri vb.
    st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">Exploring Data Analysis (EDA)</p>', unsafe_allow_html=True)
    st.image("https://data-flair.training/blogs/wp-content/uploads/sites/2/2021/10/iris-flower.webp", use_column_width=True)
    
    plt.figure(figsize=(8, 6))
    ax=sns.countplot(data=iris, x='species')
    ax.bar_label(ax.containers[0])
    plt.title('Species Distribution in Iris Dataset')
    plt.xlabel('Species')
    plt.ylabel('Count')
    st.pyplot(plt)
    
    
    st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: left; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">Statistical Properties of the Features</p>', unsafe_allow_html=True)
    st.dataframe(iris.describe())
    
    
    st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: left; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">Plot the Distributions of Features For Each Species</p>', unsafe_allow_html=True)
    sns.set(style="whitegrid")
    sns.pairplot(iris, hue='species', markers=["o", "s", "D"], diag_kind="kde")
    st.pyplot(plt)
    
    
    st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: left; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">Status of Outliers</p>', unsafe_allow_html=True)
    sns.set(style = "ticks")
    plt.figure(figsize = (12,10))

    plt.subplot(2,2,1)                                                 
    sns.boxplot(x = "species", y = "sepal_length", data = iris)
    plt.subplot(2,2,2)
    sns.boxplot(x = "species", y = "sepal_width", data = iris)
    plt.subplot(2,2,3)
    sns.boxplot(x = "species", y = "petal_length", data = iris)
    plt.subplot(2,2,4)
    sns.boxplot(x = "species", y = "petal_width", data = iris)
    st.pyplot(plt)
    
    
    st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: left; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">Correlation Table</p>', unsafe_allow_html=True)
    plt.figure(figsize = (10,11))
    sns.heatmap(iris.corr(), annot = True, cmap = "coolwarm")
    st.pyplot(plt)
    
elif selected_tab == "Prediction":
    st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">IRIS Flower Classification using TensorFlow</p>', unsafe_allow_html=True)
    st.image("https://miro.medium.com/v2/resize:fit:1200/1*S2GII1ul0JjbZ0YaFvDByw.jpeg", use_column_width=True)

    # Metni ortala
    st.markdown('<p style="text-align: center; font-size: 20px;">This app predicts the <strong>Iris flower</strong> type!</p>', unsafe_allow_html=True)
     
    model_option = st.sidebar.selectbox("Select Model", ["XGBoost", "Artificial Neural Network"])
    # Tahmin sekmesi içeriği
    st.sidebar.header('User Input Parameters for Prediction')
    
    def user_input_features():
        sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4, step=0.1)
        sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
        data = {'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width}
        features = pd.DataFrame(data, index=[0])
        return features
    
    df = user_input_features()

    # Load local Iris dataset
    #iris = pd.read_csv("iris.csv")

    final_xgboost = pickle.load(open("final_model_xgboost_iris", "rb"))
    final_ann = load_model('final_model_ANN_iris.h5')
    scaler_iris = pickle.load(open("scaler_iris", "rb"))

    st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: left; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">User Input parameters</p>', unsafe_allow_html=True)
    st.dataframe(df.style.format("{:.2f}"))


    if model_option == "XGBoost":
        prediction = final_xgboost.predict(df)
        prediction_proba = final_xgboost.predict_proba(df)
        
    elif model_option == "Artificial Neural Network":     
        df_scaled = scaler_iris.transform(df)
        y_pred_probabilities = final_ann.predict(df.values)
        prediction = y_pred_probabilities.argmax(axis=1)
        prediction_proba = tf.nn.softmax(y_pred_probabilities, axis=1).numpy()

    st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: left; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">Class labels and their corresponding index number</p>', unsafe_allow_html=True)
    st.write(iris.species.value_counts().index)
    st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: left; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">Prediction Probability</p>', unsafe_allow_html=True)
    st.write(prediction_proba)
    st.markdown('<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: left; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">Prediction</p>', unsafe_allow_html=True)

    # İki sütunlu düzen oluşturma
    col1, col2 = st.columns([2, 1])  # İlk sütunun genişliği 2, ikincisinin genişliği 1

    # Sol sütun: Yazı
    with col1:
        st.write("<p style='color:red; font-size:25px; text-align: center; margin-top: 150px;'>Prediction: {}</p>".format(", ".join(iris.species.value_counts().index[prediction])), unsafe_allow_html=True)

    # Sağ sütun: GIF
    with col2:
        st.markdown('<div style="display: flex; justify-content: flex-end;"><img src="https://media.tenor.com/VmR4yzFeI_AAAAAi/floral-fury-cuphead.gif" alt="GIF" width="100%" style="max-width: 200px; margin-right: 150px;"></div>', unsafe_allow_html=True)
