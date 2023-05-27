import pandas as pd
import streamlit as st
import numpy as np
import cv2
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

@st.cache(allow_output_mutation=True)
def load_data_1():
    return pd.read_csv('rfm_df.csv')
df = load_data_1()
@st.cache(allow_output_mutation=True)
def load_data_2():
    return pd.read_csv('rfm_data.csv')
rfm_data = load_data_2()
st.sidebar.title("Customer Segmentation")

main_data = pd.read_excel("Online Retail.xlsx")

status=st.sidebar.radio('',('Home','Recency','Frequency','Monetary','Custom Scatter Plot', "Total Segments and Correlation","Data Preprocessing",
							'KMeans Clustering Using RFM Variables','Principal Componet Analysis','Summary',"Next Section"))

if status =="Home":
	st.markdown("# Customer Segmentation and Sales Analysis")
if status=="Recency":
	st.markdown("## How much time has elapsed since a customer’s last activity or transaction with the brand?")
	option = st.selectbox("Select a Plot To Visualize", ('DistPlot',))
	if option=="DistPlot":
		plt.figure(figsize=(8,5))
		sns.distplot(df.Recency, kde=False, rug=True)
		st.pyplot()



if status == "Frequency":
	st.markdown("##  How often has a customer transacted or interacted with the brand during a particular period of time?")
	option = st.selectbox("Select a Plot To Visualize", ('DistPlot',))
	if option=="DistPlot":
		plt.figure(figsize=(8,5))
		sns.distplot(df.Frequency, kde=False, rug=True)
		st.pyplot()




if status == "Monetary":
	st.markdown("## Also referred to as “monetary value,” this factor reflects how much a customer has spent with the brand during a particular period of time.")
	option = st.selectbox("Select a Plot To Visualize", ('DistPlot',))
	
	if option=="DistPlot":
		plt.figure(figsize=(8,5))
		sns.distplot(df.Monetary, kde=False, rug=True)
		st.pyplot()

d = {'variety': ['Recency', 'Frequency', 'Monetary']}
df_options = pd.DataFrame(data=d)
if status=="Custom Scatter Plot":
	col1 = st.selectbox('Which feature on X axis?', df.columns[2:5])
	col2 = st.selectbox('Which feature on Y axis?', df.columns[2:5])
	sns.scatterplot(data=df,x=col1, y=col2)
	st.pyplot()

@st.cache(allow_output_mutation=True)
def load_data_3():
    return pd.read_csv('key_segments.csv')
df_seg = load_data_3()
@st.cache(allow_output_mutation=True)
def load_data_4():
    return pd.read_csv('rfm_des.csv')
df_desc = load_data_4()
if status == "Summary":
	st.table(df_desc)
	norm = matplotlib.colors.Normalize(vmin=min(df_seg.nofpeople), vmax=max(df_seg.nofpeople))
	colors = [matplotlib.cm.Blues(norm(value)) for value in df_seg.nofpeople]

	fig = plt.gcf()
	ax = fig.add_subplot()
	fig.set_size_inches(35, 20)


	squarify.plot(label=df_seg.segment,sizes=df_seg.nofpeople, color = colors, alpha=.8, text_kwargs={'fontsize':44}, edgecolor="white", linewidth=8)
	plt.title("Customer Segments",fontsize=42,fontweight="bold")

	plt.axis('off')
	st.pyplot()

@st.cache(allow_output_mutation=True)
def load_data_5():
    return pd.read_csv('rfm_segmentation.csv')
rfm_table = load_data_5()

if status =="Total Segments and Correlation":
	st.header("Numbers Of Customer In Each RFM Segments")
	grouped_by_rfmscore = rfm_table.groupby(['RFMScore']).size().reset_index(name='count')
	grouped_by_rfmscore = grouped_by_rfmscore.sort_values(by=['count'])
	data = [go.Bar(x=grouped_by_rfmscore['RFMScore'], y=grouped_by_rfmscore['count'])]

	layout = go.Layout(
		xaxis=go.layout.XAxis(
			title=go.layout.xaxis.Title(
				text='RFM Segment'
			)
		),
		yaxis=go.layout.YAxis(
			title=go.layout.yaxis.Title(
				text='Number of Customers'
			)
		)
	)

	fig = go.Figure(data=data, layout=layout)
	st.plotly_chart(fig)
	st.header("Correlation Heatmap of RFM Variables")
	sns.heatmap(rfm_data.corr())
	st.pyplot()

log_data = pd.read_csv("log_data.csv")
rfm_data = pd.read_csv("rfm_data.csv")
if status == "Data Preprocessing":
	st.header("Data Before Normalization")
	scatter_matrix(rfm_data, alpha = 0.3, figsize = (11,5), diagonal = 'kde');
	st.pyplot()
	st.header("Data After Normalization")
	scatter_matrix(log_data, alpha = 0.2, figsize = (11,5), diagonal = 'kde');
	st.pyplot()
	
X = matrix = log_data.to_numpy()
max = 0
max_n = 2
if status == "KMeans Clustering Using RFM Variables":
	st.header("KMeans Clustering Using Normalized RFM Variables")
	if st.button("Get Best Cluster Count And Silhouette Score"):
		for n in range(2,10):
			kmeans = KMeans(init='k-means++', n_clusters = n, n_init=100)
			kmeans.fit(matrix)
			clusters = kmeans.predict(matrix)
			silhouette_avg = silhouette_score(matrix, clusters)
			if(silhouette_avg>max):
				max = silhouette_avg
				max_n = n
		st.write("The Highest Silhouette Score is", max)
		st.write("The Best Number of Cluster is ", max_n)
	n_clusters = st.slider("Choose The Number Of Cluster To Be Made", min_value=2, max_value=10,key='n_clusters')
	if st.button("Calculate and Plot"):
		fig, (ax1, ax2) = plt.subplots(1, 2)
		fig.set_size_inches(18, 7)
		ax1.set_xlim([-0.1, 1])
		ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
		clusterer = KMeans(n_clusters=n_clusters, random_state=10)
		cluster_labels = clusterer.fit_predict(X)
		silhouette_avg = silhouette_score(X, cluster_labels)
		st.write("For The Number Of Clusters =", n_clusters,
			  "The Average Silhouette Score is :", silhouette_avg)

		sample_silhouette_values = silhouette_samples(X, cluster_labels)

		y_lower = 10
		for i in range(n_clusters):

			ith_cluster_silhouette_values = \
				sample_silhouette_values[cluster_labels == i]

			ith_cluster_silhouette_values.sort()

			size_cluster_i = ith_cluster_silhouette_values.shape[0]
			y_upper = y_lower + size_cluster_i

			color = cm.nipy_spectral(float(i) / n_clusters)
			ax1.fill_betweenx(np.arange(y_lower, y_upper),
							  0, ith_cluster_silhouette_values,
							  facecolor=color, edgecolor=color, alpha=0.7)


			ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))


			y_lower = y_upper + 10  

		ax1.set_title("The silhouette plot for the various clusters.")
		ax1.set_xlabel("The silhouette coefficient values")
		ax1.set_ylabel("Cluster label")


		ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

		ax1.set_yticks([])
		ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


		colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
		ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
					c=colors, edgecolor='k')


		centers = clusterer.cluster_centers_
		ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
					c="white", alpha=1, s=200, edgecolor='k')

		for i, c in enumerate(centers):
			ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
						s=50, edgecolor='k')

		ax2.set_title("Visualization Of The Cluster")


		plt.suptitle(("Silhouette Analysis for KMeans For Normalized RFM Data"
					  "With n_clusters = %d" % n_clusters),
					 fontsize=14, fontweight='bold')
		st.pyplot(width=1000, height=1000)

if status == "Principal Componet Analysis":
	st.markdown("## Principal Component Analysis With Recency, Frequency, Monetary")
	image = cv2.imread('PCA.jpg')
	st.image(image)
# st.sidebar.title("Predictive Analysis")
# if st.sidebar.checkbox("Analysis Details", False):
# 	st.header("Modelling Details")
# 	st.markdown(
# 		"## This model is trained on the RFM variable and it classifies a customer into 'Active' and 'Passive'. So company can focus on only the Passive Customers to market")

# def plot_metrics(metrics_list):
# 		if 'Confusion Matrix' in metrics_list:
# 			st.subheader("Confusion Matrix")
# 			cm = confusion_matrix(y_test, y_pred)
# 			fig,ax = plt.subplots()
# 			sns.heatmap(cm, annot = True)
# 			ax.figure.savefig('file.png')
# 			st.pyplot(fig)
# 		if 'Precision-Recall Curve' in metrics_list:
# 			st.subheader('Precision-Recall Curve')
			
# 			precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

# 			fig = px.area(
# 				x=recall, y=precision,
# 				title=f'Precision-Recall Curve (AUC={auc(precision, recall):.4f})',
# 				labels=dict(x='Recall', y='Precision'),
# 				width=700, height=500
# 				)
# 			fig.add_shape(
# 				type='line', line=dict(dash='dash'),
# 				x0=0, x1=1, y0=1, y1=0
# 				)
# 			fig.update_yaxes(scaleanchor="x", scaleratio=1)
# 			fig.update_xaxes(constrain='domain')
# 			st.write(fig)
			
			
# 		if 'ROC Curve' in metrics_list:
# 			fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# 			fig = px.area(
# 			   x=fpr, y=tpr,
# 			   title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
# 			   labels=dict(x='False Positive Rate', y='True Positive Rate'),
# 			   width=700, height=500
# 			   )       
# 			fig.add_shape(
# 				type='line', line=dict(dash='dash'),
# 				x0=0, x1=1, y0=0, y1=1
# 				)

# 			fig.update_yaxes(scaleanchor="x", scaleratio=1)
# 			fig.update_xaxes(constrain='domain')
# 			st.write(fig)
		
# 		if 'Training and Test accuracies' in metrics_list:
# 			mal_train_X = X_train[y_train==0]
# 			mal_train_y = y_train[y_train==0]
# 			ben_train_X = X_train[y_train==1]
# 			ben_train_y = y_train[y_train==1]
			
# 			mal_test_X = x_test[y_test==0]
# 			mal_test_y = y_test[y_test==0]
# 			ben_test_X = x_test[y_test==1]
# 			ben_test_y = y_test[y_test==1]
			
# 			scores = [model.score(mal_train_X, mal_train_y), model.score(ben_train_X, ben_train_y), model.score(mal_test_X, mal_test_y), model.score(ben_test_X, ben_test_y)]

# 			fig,ax = plt.subplots()
		
# 	# Plot the scores as a bar chart
# 			bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

# 	# directly label the score onto the bars
# 			for bar in bars:
# 				height = bar.get_height()
# 				plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), ha='center', color='w', fontsize=11)

# 	# remove all the ticks (both axes), and tick labels on the Y axis
# 			plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

# 	# remove the frame of the chart
# 			for spine in plt.gca().spines.values():
# 				spine.set_visible(False)

# 			plt.xticks([0,1,2,3], ['Active Customers\nTraining', 'Passive Customers\nTraining', 'Active Customers\nTest', 'Passive Customers\nTest'], alpha=0.8);
# 			plt.title('Training and Test Accuracies for Active and Passive', alpha=0.8)
# 			ax.xaxis.set_tick_params(length=0)
# 			ax.yaxis.set_tick_params(length=0)
# 			ax.figure.savefig('file1.png')
# 			st.pyplot(fig)
# #------------------------------------------------------------------------------------------------------------------------------------------------#
# st.sidebar.subheader("Choose Classifier")
# classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", 'Decision Tree', 'Gaussian Naive Bayes'))
# if st.sidebar.checkbox("Show X_train/Y_train", False):
#             st.subheader('X_train')
#             st.dataframe(X_train)
#             st.subheader('Y_train')
#             st.dataframe(y_train)
# if classifier == 'Support Vector Machine (SVM)':
# 	st.sidebar.subheader("Model Hyperparameters")
# 			#choose parameters
# 	C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
# 	kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
# 	gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

# 	metrics = st.sidebar.multiselect("Metrics to Plot", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve','Training and Test accuracies'))
		
# 	if st.sidebar.button("Classify", key='classify'):
# 		st.subheader("Support Vector Machine (SVM) Results")
# 		model = SVC(C=C, kernel=kernel, gamma=gamma)
# 		model.fit(X_train, y_train)
# 		accuracy = model.score(x_test, y_test)
# 		y_pred = model.predict(x_test)
# 		st.write("Accuracy: ", accuracy.round(2))
# 		st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
# 		st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
# 		plot_metrics(metrics)

# if classifier == 'Logistic Regression':
# 	st.sidebar.subheader("Model Hyperparameters")
# 	C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
# 	max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
			
# 	metrics = st.sidebar.multiselect("Metrics to Plot", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve','Training and Test accuracies'))
			
# 	if st.sidebar.button("Classify", key='classify'):
# 		st.subheader("Logistic Regression Results")
# 		model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
# 		model.fit(X_train, y_train)
# 		accuracy = model.score(x_test, y_test)
# 		y_pred = model.predict(x_test)
# 		st.write("Accuracy: ", accuracy.round(2))
# 		st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
# 		st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
# 		plot_metrics(metrics)
				
# if classifier == 'Decision Tree':
# 	st.sidebar.subheader("Model Hyperparameters")
			
# 	max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
# 	criterion = st.sidebar.radio("Criterion", ("gini", "entropy"), key='criterion')
# 	splitter = st.sidebar.radio("Splitter", ("best", "random"), key='splitter')
# 	metrics = st.sidebar.multiselect("Metrics to Plot", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve','Training and Test accuracies'))
			
# 	if st.sidebar.button("Classify", key='classify'):
# 		st.subheader("Decision Tree Results")
# 		model = DecisionTreeClassifier(max_depth= max_depth, criterion= criterion, splitter= splitter )
# 		model.fit(X_train, y_train)
# 		accuracy = model.score(x_test, y_test)
# 		y_pred = model.predict(x_test)
# 		st.write("Accuracy: ", accuracy.round(2))
# 		st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
# 		st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
# 		plot_metrics(metrics)
				
# if classifier == 'Gaussian Naive Bayes':
# 	st.sidebar.subheader("Model Hyperparameters")
		
		
# 	metrics = st.sidebar.multiselect("Metrics to Plot", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve','Training and Test accuracies'))

# 	if st.sidebar.button("Classify", key='classify'):
# 		st.subheader("Gaussian Naive Bayes Results")
# 		model = GaussianNB()
# 		model.fit(X_train, y_train)
# 		accuracy = model.score(x_test, y_test)
# 		y_pred = model.predict(x_test)
# 		st.write("Accuracy: ", accuracy.round(2))
# 		st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
# 		st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
# 		plot_metrics(metrics)     

#--------------------------------------------------------------------------------------------------------------------------------------------------------#
