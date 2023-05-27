import pandas as pd
import streamlit as st
import numpy as np

import time, warnings
import datetime as dt
import matplotlib
import matplotlib.cm as cm

import sklearn.cluster as cluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import plotly.graph_objs as go
import squarify
import plotly.express as px

from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Paired12, Category20_20
from bokeh.transform import factor_cmap

warnings.filterwarnings("ignore")
X = np.load('X.npy', allow_pickle=True)
Y = np.load("Y.npy", allow_pickle=True)
from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7)

class_names = ["Active", "Passive"]


st.sidebar.title("Predictive Analysis")
if st.sidebar.checkbox("Analysis Details", False):
	st.header("Modelling Details")
	st.markdown(
		"## This model is trained on the RFM variable and it classifies a customer into 'Active' and 'Passive'. So company can focus on only the Passive Customers to market")

def plot_metrics(metrics_list):
		if 'Confusion Matrix' in metrics_list:
			st.subheader("Confusion Matrix")
			cm = confusion_matrix(y_test, y_pred)
			fig,ax = plt.subplots()
			sns.heatmap(cm, annot = True)
			ax.figure.savefig('file.png')
			st.pyplot(fig)
		if 'Precision-Recall Curve' in metrics_list:
			st.subheader('Precision-Recall Curve')
			
			precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

			fig = px.area(
				x=recall, y=precision,
				title=f'Precision-Recall Curve (AUC={auc(precision, recall):.4f})',
				labels=dict(x='Recall', y='Precision'),
				width=700, height=500
				)
			fig.add_shape(
				type='line', line=dict(dash='dash'),
				x0=0, x1=1, y0=1, y1=0
				)
			fig.update_yaxes(scaleanchor="x", scaleratio=1)
			fig.update_xaxes(constrain='domain')
			st.write(fig)
			
			
		if 'ROC Curve' in metrics_list:
			fpr, tpr, thresholds = roc_curve(y_test, y_pred)

			fig = px.area(
			   x=fpr, y=tpr,
			   title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
			   labels=dict(x='False Positive Rate', y='True Positive Rate'),
			   width=700, height=500
			   )       
			fig.add_shape(
				type='line', line=dict(dash='dash'),
				x0=0, x1=1, y0=0, y1=1
				)

			fig.update_yaxes(scaleanchor="x", scaleratio=1)
			fig.update_xaxes(constrain='domain')
			st.write(fig)
		
		if 'Training and Test accuracies' in metrics_list:
			mal_train_X = X_train[y_train==0]
			mal_train_y = y_train[y_train==0]
			ben_train_X = X_train[y_train==1]
			ben_train_y = y_train[y_train==1]
			
			mal_test_X = x_test[y_test==0]
			mal_test_y = y_test[y_test==0]
			ben_test_X = x_test[y_test==1]
			ben_test_y = y_test[y_test==1]
			
			scores = [model.score(mal_train_X, mal_train_y), model.score(ben_train_X, ben_train_y), model.score(mal_test_X, mal_test_y), model.score(ben_test_X, ben_test_y)]

			fig,ax = plt.subplots()
		
	# Plot the scores as a bar chart
			bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

	# directly label the score onto the bars
			for bar in bars:
				height = bar.get_height()
				plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), ha='center', color='w', fontsize=11)

	# remove all the ticks (both axes), and tick labels on the Y axis
			plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

	# remove the frame of the chart
			for spine in plt.gca().spines.values():
				spine.set_visible(False)

			plt.xticks([0,1,2,3], ['Active Customers\nTraining', 'Passive Customers\nTraining', 'Active Customers\nTest', 'Passive Customers\nTest'], alpha=0.8);
			plt.title('Training and Test Accuracies for Active and Passive', alpha=0.8)
			ax.xaxis.set_tick_params(length=0)
			ax.yaxis.set_tick_params(length=0)
			ax.figure.savefig('file1.png')
			st.pyplot(fig)
#------------------------------------------------------------------------------------------------------------------------------------------------#
st.sidebar.subheader("Choose Classifier")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", 'Decision Tree', 'Gaussian Naive Bayes'))
if st.sidebar.checkbox("Show X_train/Y_train", False):
            st.subheader('X_train')
            st.dataframe(X_train)
            st.subheader('Y_train')
            st.dataframe(y_train)
if classifier == 'Support Vector Machine (SVM)':
	st.sidebar.subheader("Model Hyperparameters")
			#choose parameters
	C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
	kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
	gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

	metrics = st.sidebar.multiselect("Metrics to Plot", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve','Training and Test accuracies'))
		
	if st.sidebar.button("Classify", key='classify'):
		st.subheader("Support Vector Machine (SVM) Results")
		model = SVC(C=C, kernel=kernel, gamma=gamma)
		model.fit(X_train, y_train)
		accuracy = model.score(x_test, y_test)
		y_pred = model.predict(x_test)
		st.write("Accuracy: ", accuracy.round(2))
		st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
		st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
		plot_metrics(metrics)

if classifier == 'Logistic Regression':
	st.sidebar.subheader("Model Hyperparameters")
	C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
	max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
			
	metrics = st.sidebar.multiselect("Metrics to Plot", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve','Training and Test accuracies'))
			
	if st.sidebar.button("Classify", key='classify'):
		st.subheader("Logistic Regression Results")
		model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
		model.fit(X_train, y_train)
		accuracy = model.score(x_test, y_test)
		y_pred = model.predict(x_test)
		st.write("Accuracy: ", accuracy.round(2))
		st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
		st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
		plot_metrics(metrics)
				
if classifier == 'Decision Tree':
	st.sidebar.subheader("Model Hyperparameters")
			
	max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
	criterion = st.sidebar.radio("Criterion", ("gini", "entropy"), key='criterion')
	splitter = st.sidebar.radio("Splitter", ("best", "random"), key='splitter')
	metrics = st.sidebar.multiselect("Metrics to Plot", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve','Training and Test accuracies'))
			
	if st.sidebar.button("Classify", key='classify'):
		st.subheader("Decision Tree Results")
		model = DecisionTreeClassifier(max_depth= max_depth, criterion= criterion, splitter= splitter )
		model.fit(X_train, y_train)
		accuracy = model.score(x_test, y_test)
		y_pred = model.predict(x_test)
		st.write("Accuracy: ", accuracy.round(2))
		st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
		st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
		plot_metrics(metrics)
				
if classifier == 'Gaussian Naive Bayes':
	st.sidebar.subheader("Model Hyperparameters")
		
		
	metrics = st.sidebar.multiselect("Metrics to Plot", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve','Training and Test accuracies'))

	if st.sidebar.button("Classify", key='classify'):
		st.subheader("Gaussian Naive Bayes Results")
		model = GaussianNB()
		model.fit(X_train, y_train)
		accuracy = model.score(x_test, y_test)
		y_pred = model.predict(x_test)
		st.write("Accuracy: ", accuracy.round(2))
		st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
		st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
		plot_metrics(metrics)     

#--------------------------------------------------------------------------------------------------------------------------------------------------------#
