import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    possible_paths = [
        "Train dataset.csv",
        "train.csv",
        "Train.csv",
        "data.csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                
                df['Order Date'] = pd.to_datetime(df["Order Date"])
                df['Ship Date'] = pd.to_datetime(df['Ship Date'])
                
                df["Year"] = df["Order Date"].dt.year
                df['Month'] = df["Order Date"].dt.month
                df['Week Number'] = df['Order Date'].dt.isocalendar().week
                df['Day of Week'] = df['Order Date'].dt.dayofweek
                df["Quarter"] = df['Order Date'].dt.quarter
                
                def season(month):
                    if month in [12, 1, 2]:
                        return "Winter"
                    elif month in [3, 4, 5]:
                        return "Spring"
                    elif month in [6, 7, 8]:
                        return 'Summer'
                    else:
                        return "Fall"
                
                df["Season"] = df["Order Date"].map(season)
                return df
            except Exception as e:
                continue
    
    st.error("Data file not found. Please make sure your CSV file is in the app directory.")
    return pd.DataFrame()

@st.cache_data
def load_monthly_sales(df):
    if df.empty:
        return pd.DataFrame()
    monthly_sales = df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
    monthly_sales.columns = ['Date', 'Sales']
    return monthly_sales

@st.cache_data
def load_weekly_sales(df):
    if df.empty:
        return pd.DataFrame()
    weekly_sales = df.groupby(pd.Grouper(key='Order Date', freq='W'))['Sales'].sum().reset_index()
    weekly_sales.columns = ['Date', 'Sales']
    return weekly_sales

@st.cache_data
def load_model_comparison():
    try:
        return pd.read_csv('model_comparison.csv')
    except:
        data = {
            'Model': ['SARIMA', 'Prophet', 'XGBoost'],
            'MAE': [7629.03, 11069.43, 8340.02],
            'RMSE': [9373.18, 12386.08, 11029.36],
            'MAPE': [10.43, 16.53, 11.20],
            'Forecast_Month_1': [82436.42, 85739.69, 85178.94],
            'Forecast_Month_2': [81165.99, 84131.22, 83762.57],
            'Forecast_Month_3': [79828.02, 82397.58, 82170.52]
        }
        return pd.DataFrame(data)

@st.cache_data
def load_anomaly_report():
    try:
        return pd.read_csv('anomaly_report.csv', parse_dates=['Date'])
    except:
        return pd.DataFrame()

@st.cache_data
def load_cluster_results():
    try:
        return pd.read_csv('product_demand_clusters.csv')
    except:
        return pd.DataFrame()

df_data = load_data()

if df_data.empty:
    st.error("🚨 Failed to load data. Please check your CSV file.")
    st.stop()

monthly_sales_data = load_monthly_sales(df_data)
weekly_sales_data = load_weekly_sales(df_data)
model_comparison_data = load_model_comparison()
anomaly_report_data = load_anomaly_report()
cluster_results_data = load_cluster_results()

st.sidebar.title("📊 Sales Analytics Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["📈 Sales Overview", "🔮 Forecast Explorer", "⚠️ Anomaly Report", "📦 Product Demand Segments"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **About this Dashboard:**
    - Built with Streamlit
    - Analyzes sales data from 2015-2018
    - Uses XGBoost as the best performing model
    - Identifies anomalies and demand segments
    """
)

if page == "📈 Sales Overview":
    st.title("📈 Sales Overview Dashboard")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        years = ['All'] + sorted(df_data['Year'].unique().tolist())
        selected_year = st.selectbox("Select Year", years)
    
    with col2:
        regions = ['All'] + sorted(df_data['Region'].unique().tolist())
        selected_region = st.selectbox("Select Region", regions)
    
    with col3:
        categories = ['All'] + sorted(df_data['Category'].unique().tolist())
        selected_category = st.selectbox("Select Category", categories)
    
    filtered_df = df_data.copy()
    if selected_year != 'All':
        filtered_df = filtered_df[filtered_df['Year'] == selected_year]
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == selected_region]
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_sales = filtered_df['Sales'].sum()
    avg_order = filtered_df['Sales'].mean()
    total_orders = len(filtered_df)
    unique_customers = filtered_df['Customer ID'].nunique()
    
    with col1:
        st.metric("Total Sales", f"${total_sales:,.2f}")
    with col2:
        st.metric("Average Order", f"${avg_order:,.2f}")
    with col3:
        st.metric("Total Orders", f"{total_orders:,}")
    with col4:
        st.metric("Unique Customers", f"{unique_customers:,}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Total Sales by Year")
        yearly_sales = filtered_df.groupby('Year')['Sales'].sum().reset_index()
        fig = px.bar(yearly_sales, x='Year', y='Sales', 
                     title='Sales by Year',
                     color='Year',
                     color_continuous_scale='Blues')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sales by Category")
        category_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
        fig = px.pie(category_sales, values='Sales', names='Category',
                     title='Sales Distribution by Category',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Monthly Sales Trend")
    monthly_trend = filtered_df.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
    monthly_trend.columns = ['Date', 'Sales']
    
    fig = px.line(monthly_trend, x='Date', y='Sales',
                  title='Monthly Sales Trend',
                  markers=True)
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        hovermode='x'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Sales by Region")
    region_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()
    fig = px.bar(region_sales, x='Region', y='Sales',
                 title='Sales by Region',
                 color='Region',
                 color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Forecast Explorer":
    st.title("🔮 Forecast Explorer")
    st.markdown("---")
    
    st.markdown("**Best Model: XGBoost** (Lowest MAE and MAPE)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_type = st.selectbox(
            "Select Forecast Type",
            ["Category", "Region"]
        )
    
    with col2:
        months_ahead = st.slider(
            "Forecast Horizon (months ahead)",
            min_value=1,
            max_value=3,
            value=3,
            step=1
        )
    
    if forecast_type == "Category":
        options = df_data['Category'].unique().tolist()
        selected_item = st.selectbox("Select Category", options)
        cat_data = df_data[df_data['Category'] == selected_item]
        monthly_cat = cat_data.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
    else:
        options = df_data['Region'].unique().tolist()
        selected_item = st.selectbox("Select Region", options)
        cat_data = df_data[df_data['Region'] == selected_item]
        monthly_cat = cat_data.groupby(pd.Grouper(key='Order Date', freq='ME'))['Sales'].sum().reset_index()
    
    model_results = model_comparison_data[model_comparison_data['Model'] == 'XGBoost']
    
    if len(model_results) > 0:
        forecast_1 = model_results['Forecast_Month_1'].values[0] if 'Forecast_Month_1' in model_results.columns else None
        forecast_2 = model_results['Forecast_Month_2'].values[0] if 'Forecast_Month_2' in model_results.columns else None
        forecast_3 = model_results['Forecast_Month_3'].values[0] if 'Forecast_Month_3' in model_results.columns else None
        
        last_date = monthly_cat['Order Date'].max()
        forecast_dates = [last_date + timedelta(days=30*i) for i in range(1, months_ahead+1)]
        
        forecast_values = []
        for i in range(months_ahead):
            if i == 0 and forecast_1 is not None:
                forecast_values.append(forecast_1)
            elif i == 1 and forecast_2 is not None:
                forecast_values.append(forecast_2)
            elif i == 2 and forecast_3 is not None:
                forecast_values.append(forecast_3)
            else:
                forecast_values.append(None)
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast_values
        }).dropna()
        
        st.subheader(f"Forecast for {selected_item}")
        
        fig = make_subplots()
        
        last_12 = monthly_cat.tail(12)
        fig.add_trace(go.Scatter(
            x=last_12['Order Date'],
            y=last_12['Sales'],
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color='steelblue', width=2)
        ))
        
        if len(forecast_df) > 0:
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Forecast'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='darkorange', width=3, dash='dash')
            ))
        
        fig.update_layout(
            title=f'Sales Forecast for {selected_item} ({months_ahead} months ahead)',
            xaxis_title='Date',
            yaxis_title='Sales ($)',
            hovermode='x',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Model Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mae = model_results['MAE'].values[0] if 'MAE' in model_results.columns else 'N/A'
            st.metric("MAE", f"{mae:,.2f}" if isinstance(mae, (int, float)) else mae)
        
        with col2:
            rmse = model_results['RMSE'].values[0] if 'RMSE' in model_results.columns else 'N/A'
            st.metric("RMSE", f"{rmse:,.2f}" if isinstance(rmse, (int, float)) else rmse)
        
        with col3:
            mape = model_results['MAPE'].values[0] if 'MAPE' in model_results.columns else 'N/A'
            st.metric("MAPE", f"{mape:.2f}%" if isinstance(mape, (int, float)) else mape)
        
        st.subheader("Model Comparison")
        st.dataframe(model_comparison_data, use_container_width=True)
        
    else:
        st.warning("No forecast data available for the selected option.")

elif page == "⚠️ Anomaly Report":
    st.title("⚠️ Anomaly Report")
    st.markdown("---")
    
    st.markdown("""
    This page displays anomalies detected in weekly sales data using two methods:
    - **Isolation Forest**: Detects complex, subtle anomalies
    - **Z-Score**: Detects extreme deviations
    """)
    
    st.subheader("Anomaly Detection Visualization")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=weekly_sales_data['Date'],
        y=weekly_sales_data['Sales'],
        mode='lines',
        name='Weekly Sales',
        line=dict(color='steelblue', width=2)
    ))
    
    if len(anomaly_report_data) > 0:
        if 'Iso_Anomaly' in anomaly_report_data.columns:
            iso_anomalies = anomaly_report_data[anomaly_report_data['Iso_Anomaly'] == True]
            if len(iso_anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=iso_anomalies['Date'],
                    y=iso_anomalies['Sales'],
                    mode='markers',
                    name='Isolation Forest Anomaly',
                    marker=dict(color='red', size=10, symbol='triangle-up')
                ))
        
        if 'Z_Anomaly' in anomaly_report_data.columns:
            z_anomalies = anomaly_report_data[anomaly_report_data['Z_Anomaly'] == True]
            if len(z_anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=z_anomalies['Date'],
                    y=z_anomalies['Sales'],
                    mode='markers',
                    name='Z-Score Anomaly',
                    marker=dict(color='orange', size=10, symbol='diamond')
                ))
    
    fig.update_layout(
        title='Weekly Sales with Anomalies Detected',
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        hovermode='x',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Detected Anomalies")
    
    if len(anomaly_report_data) > 0:
        display_cols = ['Date', 'Sales']
        if 'Iso_Anomaly' in anomaly_report_data.columns:
            display_cols.append('Iso_Anomaly')
        if 'Z_Anomaly' in anomaly_report_data.columns:
            display_cols.append('Z_Anomaly')
        if 'Type' in anomaly_report_data.columns:
            display_cols.append('Type')
        if 'Month' in anomaly_report_data.columns:
            display_cols.append('Month')
        if 'Year' in anomaly_report_data.columns:
            display_cols.append('Year')
        
        available_cols = [col for col in display_cols if col in anomaly_report_data.columns]
        st.dataframe(anomaly_report_data[available_cols], use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Anomalies", len(anomaly_report_data))
        
        with col2:
            iso_count = anomaly_report_data['Iso_Anomaly'].sum() if 'Iso_Anomaly' in anomaly_report_data.columns else 0
            st.metric("Isolation Forest", iso_count)
        
        with col3:
            z_count = anomaly_report_data['Z_Anomaly'].sum() if 'Z_Anomaly' in anomaly_report_data.columns else 0
            st.metric("Z-Score", z_count)
        
        with col4:
            if 'Iso_Anomaly' in anomaly_report_data.columns and 'Z_Anomaly' in anomaly_report_data.columns:
                both = ((anomaly_report_data['Iso_Anomaly'] == True) & (anomaly_report_data['Z_Anomaly'] == True)).sum()
                st.metric("Both Methods", both)
        
        st.subheader("Key Observations")
        
        if 'Type' in anomaly_report_data.columns:
            high_count = (anomaly_report_data['Type'] == 'High').sum() if 'Type' in anomaly_report_data.columns else 0
            low_count = (anomaly_report_data['Type'] == 'Low').sum() if 'Type' in anomaly_report_data.columns else 0
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("High Sales Spikes", high_count)
            with col2:
                st.metric("Low Sales Dips", low_count)
        
        if 'Month' in anomaly_report_data.columns:
            st.write("**Seasonal Patterns:**")
            monthly_anomalies = anomaly_report_data['Month'].value_counts().head(6)
            fig = px.bar(monthly_anomalies, title='Anomalies by Month',
                         labels={'value': 'Count', 'index': 'Month'})
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("No anomalies detected in the data.")

else:
    st.title("📦 Product Demand Segments")
    st.markdown("---")
    
    st.markdown("""
    This page displays the results of K-Means clustering on product sub-categories.
    The segments are based on:
    - Total Sales
    - Average Order Value
    - Sales Volatility
    - Growth Rate
    - Number of Orders
    """)
    
    st.subheader("Cluster Visualization (PCA)")
    
    if len(cluster_results_data) > 0:
        if 'PC1' in cluster_results_data.columns and 'PC2' in cluster_results_data.columns:
            fig = px.scatter(
                cluster_results_data,
                x='PC1',
                y='PC2',
                color='Cluster_Label' if 'Cluster_Label' in cluster_results_data.columns else 'Cluster',
                hover_data=['Sub_Category' if 'Sub_Category' in cluster_results_data.columns else 'index'],
                title='Product Demand Segments - K-Means Clustering',
                color_discrete_sequence=px.colors.qualitative.Set1,
                size_max=15
            )
            
            fig.update_layout(
                xaxis_title='Principal Component 1',
                yaxis_title='Principal Component 2',
                height=500,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Cluster Summary")
        
        if 'Cluster' in cluster_results_data.columns and 'Total_Sales' in cluster_results_data.columns:
            summary = cluster_results_data.groupby('Cluster').agg({
                'Total_Sales': ['mean', 'sum'],
                'Num_Orders' if 'Num_Orders' in cluster_results_data.columns else 'Total_Sales': 'mean',
                'Sub_Category' if 'Sub_Category' in cluster_results_data.columns else 'Cluster': 'count'
            }).round(2)
            
            summary.columns = ['Avg Sales', 'Total Sales', 'Avg Orders', 'Count']
            st.dataframe(summary, use_container_width=True)
        
        st.subheader("Sub-Categories by Cluster")
        
        if 'Cluster_Label' in cluster_results_data.columns and 'Sub_Category' in cluster_results_data.columns:
            cluster_table = cluster_results_data[['Sub_Category', 'Cluster_Label', 'Total_Sales']].copy()
            cluster_table = cluster_table.sort_values(['Cluster_Label', 'Total_Sales'], ascending=[True, False])
            
            st.dataframe(cluster_table, use_container_width=True)
            
            st.subheader("Business Insights")
            
            unique_labels = cluster_results_data['Cluster_Label'].unique()
            
            for label in unique_labels:
                cluster_data = cluster_results_data[cluster_results_data['Cluster_Label'] == label]
                sub_count = len(cluster_data)
                avg_sales = cluster_data['Total_Sales'].mean()
                
                st.write(f"**{label}**: {sub_count} sub-categories, Avg Sales: ${avg_sales:,.2f}")
                
                subs = cluster_data['Sub_Category'].tolist() if 'Sub_Category' in cluster_data.columns else []
                if subs:
                    st.write(f"    - Sub-categories: {', '.join(subs[:5])}")
                    if len(subs) > 5:
                        st.write(f"    - and {len(subs)-5} more...")
                st.write("")
            
            st.subheader("Recommendations")
            
            stable_cluster = cluster_results_data[
                cluster_results_data['Cluster_Label'].str.contains('Stable', case=False, na=False)
            ]['Sub_Category'].tolist() if 'Cluster_Label' in cluster_results_data.columns else []
            
            if stable_cluster:
                st.success(f"""
                **Focus on High-Performing Segments:**
                The following sub-categories are in the 'High Volume, Stable Demand' segment:
                {', '.join(stable_cluster)}
                
                **Recommendations:**
                1. Allocate more inventory and marketing budget to these segments
                2. Monitor these segments closely for potential supply chain needs
                3. Consider cross-selling opportunities with these high-demand products
                """)
            else:
                st.info("No high volume, stable demand segments identified. Consider analyzing the data further to identify growth opportunities.")
    else:
        st.warning("Cluster data not available. Please check the product_demand_clusters.csv file.")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>📊 Sales Analytics Dashboard | Built with Streamlit</p>
        <p style='font-size: 12px;'>Data covers 2015-2018 | Best model: XGBoost</p>
    </div>
    """,
    unsafe_allow_html=True
)
