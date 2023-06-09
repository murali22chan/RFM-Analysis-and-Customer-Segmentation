# RFM-Analysis-and-Customer-Segmentation
Developed dashboards to visualize EDA, RFM analysis and Customer Segmentation (RFM quartiles and ML) leveraging previous customer's purchase history data.

 • **data/** contains the entire data required for the analysis. <br>
 • **notebooks/** contains the notebooks for EDA, RFM analysis and Customer Segmentation. <br>
 • **scripts/** contatins the scripts to run the dashboards. <br>

## Usage
1. Git Clone the Repo
```python
git clone https://github.com/murali22chan/RFM-Analysis-and-Customer-Segmentation.git
```
2. Install the required packages.
```python 
pip install -r requirements.txt 
```
3. To run the EDA analysis dashboard
```python 
streamlit run sales_app.py
```
4. To run the RFM analysis and Customer Segmentation dashboard
```python 
streamlit run customer_app.py
```
5. To run the ML predictive app
```python 
streamlit run predictive_app.py
```
