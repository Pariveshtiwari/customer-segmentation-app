# Universal Customer Segmentation App

This Streamlit application performs **K-Means based customer segmentation**  
on ANY CSV dataset containing numeric columns.

### Features
- Upload any dataset (.csv)
- Auto-detect numeric columns
- Choose features to cluster
- Elbow method visualization
- K-Means clustering
- Cluster visualization with centroids
- Download segmented dataset

### How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
