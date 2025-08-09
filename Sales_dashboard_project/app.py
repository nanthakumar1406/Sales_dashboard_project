import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from flask import Flask, render_template, send_file
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import numpy as np

app = Flask(__name__)

# Load and preprocess data
def load_and_preprocess_data():
    try:
        df = pd.read_csv(r"C:/Users/Lenova/OneDrive/ML/Sales_dashboard_project/SuperStore_Sales_Dataset.csv")

        # Convert date columns to datetime
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce', dayfirst=True)
        df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce', dayfirst=True)
    
        # List of columns to convert to categorical
        categorical_columns = [
            'Order ID', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country',
            'City', 'State', 'Region', 'Product ID', 'Category', 'Sub-Category',
            'Product Name', 'Payment Mode'
        ]
    
        # Convert to 'category' dtype
        for col in categorical_columns:
            df[col] = df[col].astype('category')
    
        # Drop unnecessary columns
        df.drop(columns=['Returns', 'ind1', 'ind2'], inplace=True, errors='ignore')
        
        return df

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

df = load_and_preprocess_data()

# Helper function to convert matplotlib plot to base64 for HTML
def plot_to_base64(plt):
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

@app.route('/')
def index():
    if df is None:
        return "<h2>Error: Dataset could not be loaded.</h2>"
    rows, cols = df.shape
    categorical_columns = df.select_dtypes(include='category').columns.tolist()
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    duplicate_count = df.duplicated().sum()
    df_head = df.head().to_html(classes='table table-striped', index=False)
    df_summary = df.describe().to_html(classes='table table-bordered')
    
    # df.info() capture to string
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info = buffer.getvalue()

    return render_template(
        'index.html',
        rows=rows,
        cols=cols,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        duplicate_count=duplicate_count,
        df_head=df_head,
        df_info=df_info,
        df_summary=df_summary
    )

@app.route('/visualizations')
def visualizations():
    if df is None:
        return "<h2>Error: Dataset could not be loaded. Check file path and format.</h2>"
    # Create plots
    plots = {}
    
    # Sales Distribution
    plt.figure(figsize=(8,5))
    sns.histplot(df['Sales'], bins=50, kde=True)
    plt.title('Sales Distribution')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    plots['sales_dist'] = plot_to_base64(plt)
    
    # Profit Distribution
    plt.figure(figsize=(8,5))
    sns.histplot(df['Profit'], bins=50, kde=True, color='orange')
    plt.title('Profit Distribution')
    plt.xlabel('Profit')
    plt.ylabel('Frequency')
    plots['profit_dist'] = plot_to_base64(plt)
    
    # Sales by Category
    plt.figure(figsize=(8,5))
    sns.boxplot(x='Category', y='Sales', data=df)
    plt.title('Sales by Category')
    plt.ylabel('Sales')
    plots['sales_category'] = plot_to_base64(plt)
    
    # Total Sales by Region
    region_sales = df.groupby('Region')['Sales'].sum().reset_index()
    plt.figure(figsize=(8,5))
    sns.barplot(data=region_sales, x='Region', y='Sales')
    plt.title('Total Sales by Region')
    plt.ylabel('Total Sales')
    plots['sales_region'] = plot_to_base64(plt)
    
    # Profit vs Sales by Category
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='Sales', y='Profit', hue='Category', alpha=0.6, edgecolor='w')
    plt.title('Profit vs Sales by Category')
    plt.xlabel('Sales')
    plt.ylabel('Profit')
    plt.legend()
    plots['profit_vs_sales'] = plot_to_base64(plt)
    
    return render_template('visualizations.html', plots=plots)

@app.route('/models')
def models():
    # Prepare data for regression
    X_reg = df.drop(columns=['Row ID+O6G3A1:R6', 'Order ID', 'Customer ID', 'Customer Name',
                            'Order Date', 'Ship Date', 'Product ID', 'Product Name', 'Profit'])
    y_reg = df['Profit']
    
    X_reg_encoded = pd.get_dummies(X_reg, drop_first=True)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg_encoded, y_reg, test_size=0.2, random_state=42)
    
    # Random Forest Regressor
    rf_reg = RandomForestRegressor(random_state=42)
    rf_reg.fit(X_train_reg, y_train_reg)
    y_pred_rf = rf_reg.predict(X_test_reg)
    rmse_rf = np.sqrt(mean_squared_error(y_test_reg, y_pred_rf))
    r2_rf = r2_score(y_test_reg, y_pred_rf)
    
    # Feature Importance for RF Regressor
    plt.figure(figsize=(10, 5))
    importances = rf_reg.feature_importances_
    indices = np.argsort(importances)[-10:]  # top 10 features
    plt.title("Top 10 Feature Importances - Random Forest Regressor")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [X_reg_encoded.columns[i] for i in indices])
    plt.xlabel("Relative Importance")
    rf_reg_plot = plot_to_base64(plt)
    
    # Decision Tree Regressor
    dt_reg = DecisionTreeRegressor(random_state=42)
    dt_reg.fit(X_train_reg, y_train_reg)
    y_pred_dt = dt_reg.predict(X_test_reg)
    rmse_dt = np.sqrt(mean_squared_error(y_test_reg, y_pred_dt))
    r2_dt = r2_score(y_test_reg, y_pred_dt)
    
    # Decision Tree Visualization (first 3 levels)
    plt.figure(figsize=(20, 10))
    plot_tree(dt_reg, feature_names=X_train_reg.columns, filled=True, max_depth=3, fontsize=10)
    plt.title("Decision Tree Regressor (Top Levels)")
    dt_tree_plot = plot_to_base64(plt)
    
    # Actual vs Predicted for Decision Tree
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_reg, y_pred_dt, alpha=0.5, color='teal')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted - Decision Tree Regressor")
    plt.plot([y_test_reg.min(), y_test_reg.max()],
             [y_test_reg.min(), y_test_reg.max()], 'r--')  # ideal line
    plt.grid(True)
    dt_scatter_plot = plot_to_base64(plt)
    
    # Classification models
    df['High_Profit'] = df['Profit'].apply(lambda x: 1 if x > 50 else 0)
    
    X_clf = df.drop(['Profit', 'Sales', 'High_Profit', 'Order ID', 'Customer ID', 'Product ID',
                    'Customer Name', 'Product Name', 'Order Date', 'Ship Date'], axis=1)
    y_clf = df['High_Profit']
    
    X_clf_encoded = pd.get_dummies(X_clf, drop_first=True)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf_encoded, y_clf, test_size=0.2, random_state=42)
    
    # Random Forest Classifier
    rf_clf = RandomForestClassifier(random_state=42)
    rf_clf.fit(X_train_clf, y_train_clf)
    y_pred_rf_clf = rf_clf.predict(X_test_clf)
    rf_clf_accuracy = accuracy_score(y_test_clf, y_pred_rf_clf)
    rf_clf_cm = confusion_matrix(y_test_clf, y_pred_rf_clf)
    rf_clf_report = classification_report(y_test_clf, y_pred_rf_clf)
    
    # Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(rf_clf_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Random Forest Classifier')
    rf_cm_plot = plot_to_base64(plt)
    
    # Feature Importance for RF Classifier
    plt.figure(figsize=(10, 6))
    importances = rf_clf.feature_importances_
    feature_names = X_train_clf.columns
    feature_importance_pairs = list(zip(feature_names, importances))
    sorted_features = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:10]
    names, scores = zip(*top_features)
    plt.barh(names[::-1], scores[::-1], color='skyblue')
    plt.xlabel("Feature Importance Score")
    plt.title("Top 10 Important Features - Random Forest Classifier")
    plt.grid(True)
    plt.tight_layout()
    rf_feature_plot = plot_to_base64(plt)
    
    return render_template('models.html',
                         rf_reg_metrics={'RMSE': rmse_rf, 'R2': r2_rf},
                         rf_reg_plot=rf_reg_plot,
                         dt_reg_metrics={'RMSE': rmse_dt, 'R2': r2_dt},
                         dt_tree_plot=dt_tree_plot,
                         dt_scatter_plot=dt_scatter_plot,
                         rf_clf_accuracy=rf_clf_accuracy,
                         rf_clf_cm=rf_clf_cm,
                         rf_clf_report=rf_clf_report,
                         rf_cm_plot=rf_cm_plot,
                         rf_feature_plot=rf_feature_plot)

if __name__ == '__main__':
    app.run(debug=True)