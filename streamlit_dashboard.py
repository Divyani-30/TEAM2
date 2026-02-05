"""
Streamlit Interactive Dashboard
Vegetable Market Intelligence Analysis

Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Vegetable Market Intelligence",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the master dataset"""
    try:
        df = pd.read_csv('outputs/master_dataset.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except:
        st.error("Please run the analytics framework first to generate the master dataset!")
        st.stop()

# Load data
df = load_data()

# Sidebar
st.sidebar.title("🥬 Market Intelligence")
st.sidebar.markdown("---")

# Date range filter
min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Category filter
categories = ['All'] + sorted(df['Category Name'].unique().tolist())
selected_category = st.sidebar.selectbox("Category", categories)

# Item filter
if selected_category != 'All':
    items = ['All'] + sorted(df[df['Category Name'] == selected_category]['Item Name'].unique().tolist())
else:
    items = ['All'] + sorted(df['Item Name'].unique().tolist())
selected_item = st.sidebar.selectbox("Item", items)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Analysis Options")
show_outliers = st.sidebar.checkbox("Show Outliers", value=False)
show_trends = st.sidebar.checkbox("Show Trend Lines", value=True)

# Filter data
if len(date_range) == 2:
    filtered_df = df[(df['Date'] >= pd.Timestamp(date_range[0])) & 
                     (df['Date'] <= pd.Timestamp(date_range[1]))]
else:
    filtered_df = df

if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['Category Name'] == selected_category]

if selected_item != 'All':
    filtered_df = filtered_df[filtered_df['Item Name'] == selected_item]

# Main content
st.title("🥬 Vegetable Market Intelligence Dashboard")
st.markdown("### Comprehensive Analytics Platform")

# Key Metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Records",
        f"{len(filtered_df):,}",
        f"{len(filtered_df)/len(df)*100:.1f}% of total"
    )

with col2:
    avg_price = filtered_df['Wholesale Price (RMB/kg)'].mean()
    st.metric(
        "Avg Price",
        f"¥{avg_price:.2f}/kg"
    )

with col3:
    avg_loss = filtered_df['Loss Rate (%)'].mean()
    st.metric(
        "Avg Loss Rate",
        f"{avg_loss:.2f}%"
    )

with col4:
    price_vol = filtered_df['Wholesale Price (RMB/kg)'].std() / avg_price
    st.metric(
        "Price Volatility",
        f"{price_vol:.2f}",
        "Coefficient of Variation"
    )

with col5:
    unique_items = filtered_df['Item Code'].nunique()
    st.metric(
        "Unique Items",
        f"{unique_items}"
    )

st.markdown("---")

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview", 
    "💰 Price Analysis", 
    "📦 Loss Analysis", 
    "📅 Time Series", 
    "🎯 Insights"
])

with tab1:
    st.header("Market Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution")
        fig = px.histogram(
            filtered_df, 
            x='Wholesale Price (RMB/kg)',
            nbins=50,
            title="Wholesale Price Distribution",
            labels={'Wholesale Price (RMB/kg)': 'Price (RMB/kg)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Loss Rate Distribution")
        fig = px.histogram(
            filtered_df,
            x='Loss Rate (%)',
            nbins=50,
            title="Loss Rate Distribution",
            color_discrete_sequence=['coral']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Category Performance")
    category_stats = filtered_df.groupby('Category Name').agg({
        'Wholesale Price (RMB/kg)': 'mean',
        'Loss Rate (%)': 'mean',
        'Item Code': 'count'
    }).round(2)
    category_stats.columns = ['Avg Price (RMB/kg)', 'Avg Loss (%)', 'Count']
    category_stats = category_stats.sort_values('Avg Price (RMB/kg)', ascending=False)
    
    fig = go.Figure(data=[
        go.Bar(name='Avg Price', x=category_stats.index, y=category_stats['Avg Price (RMB/kg)']),
        go.Bar(name='Avg Loss %', x=category_stats.index, y=category_stats['Avg Loss (%)'], yaxis='y2')
    ])
    fig.update_layout(
        title='Category Performance: Price vs Loss Rate',
        yaxis=dict(title='Avg Price (RMB/kg)'),
        yaxis2=dict(title='Avg Loss (%)', overlaying='y', side='right'),
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Price Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Most Expensive Items")
        top_items = filtered_df.groupby('Item Name')['Wholesale Price (RMB/kg)'].mean().nlargest(10).reset_index()
        fig = px.bar(
            top_items,
            y='Item Name',
            x='Wholesale Price (RMB/kg)',
            orientation='h',
            title="Top 10 Highest Priced Items"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Price vs Loss Rate")
        sample_df = filtered_df.sample(min(1000, len(filtered_df)))
        fig = px.scatter(
            sample_df,
            x='Loss Rate (%)',
            y='Wholesale Price (RMB/kg)',
            color='Category Name',
            title="Price vs Loss Rate by Category",
            hover_data=['Item Name']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Price Volatility Analysis")
    volatility = filtered_df.groupby('Item Name').agg({
        'Wholesale Price (RMB/kg)': ['mean', 'std']
    }).reset_index()
    volatility.columns = ['Item Name', 'Mean Price', 'Std Price']
    volatility['CV'] = volatility['Std Price'] / volatility['Mean Price']
    top_volatile = volatility.nlargest(15, 'CV')
    
    fig = px.bar(
        top_volatile,
        y='Item Name',
        x='CV',
        orientation='h',
        title="Top 15 Most Volatile Items (Coefficient of Variation)",
        color='CV',
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Loss Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 High Loss Items")
        top_loss = filtered_df.groupby('Item Name')['Loss Rate (%)'].mean().nlargest(10).reset_index()
        fig = px.bar(
            top_loss,
            y='Item Name',
            x='Loss Rate (%)',
            orientation='h',
            title="Top 10 Items by Loss Rate",
            color='Loss Rate (%)',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Loss Rate by Category")
        category_loss = filtered_df.groupby('Category Name')['Loss Rate (%)'].mean().reset_index()
        fig = px.bar(
            category_loss,
            x='Category Name',
            y='Loss Rate (%)',
            title="Average Loss Rate by Category",
            color='Loss Rate (%)',
            color_continuous_scale='Oranges'
        )
        fig.update_layout(height=400)
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Effective Price Impact")
    st.markdown("Effective Price = Wholesale Price × (1 + Loss Rate / 100)")
    
    sample_df = filtered_df.sample(min(500, len(filtered_df)))
    fig = px.scatter(
        sample_df,
        x='Wholesale Price (RMB/kg)',
        y='Effective Price (RMB/kg)',
        color='Loss Rate (%)',
        title="Wholesale vs Effective Price (Impact of Loss Rate)",
        hover_data=['Item Name', 'Category Name']
    )
    fig.add_trace(go.Scatter(
        x=[0, sample_df['Wholesale Price (RMB/kg)'].max()],
        y=[0, sample_df['Wholesale Price (RMB/kg)'].max()],
        mode='lines',
        name='y=x (no loss)',
        line=dict(dash='dash', color='red')
    ))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Time Series Analysis")
    
    # Monthly trends
    st.subheader("Monthly Price Trends")
    monthly = filtered_df.groupby(filtered_df['Date'].dt.to_period('M')).agg({
        'Wholesale Price (RMB/kg)': ['mean', 'std', 'min', 'max']
    }).reset_index()
    monthly.columns = ['Date', 'Mean', 'Std', 'Min', 'Max']
    monthly['Date'] = monthly['Date'].dt.to_timestamp()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly['Date'], y=monthly['Mean'],
        mode='lines+markers',
        name='Average Price',
        line=dict(width=3)
    ))
    fig.add_trace(go.Scatter(
        x=monthly['Date'], y=monthly['Max'],
        mode='lines',
        name='Max Price',
        line=dict(dash='dash'),
        opacity=0.5
    ))
    fig.add_trace(go.Scatter(
        x=monthly['Date'], y=monthly['Min'],
        mode='lines',
        name='Min Price',
        line=dict(dash='dash'),
        opacity=0.5
    ))
    fig.update_layout(
        title="Monthly Price Trends with Min/Max Range",
        xaxis_title="Date",
        yaxis_title="Price (RMB/kg)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Seasonal Patterns")
        seasonal = filtered_df.groupby(filtered_df['Date'].dt.month)['Wholesale Price (RMB/kg)'].mean().reset_index()
        seasonal.columns = ['Month', 'Avg Price']
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        seasonal['Month Name'] = seasonal['Month'].apply(lambda x: months[x-1])
        
        fig = px.bar(
            seasonal,
            x='Month Name',
            y='Avg Price',
            title="Seasonal Price Patterns",
            color='Avg Price',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Day of Week Analysis")
        dow = filtered_df.groupby(filtered_df['Date'].dt.dayofweek)['Wholesale Price (RMB/kg)'].mean().reset_index()
        dow.columns = ['DayOfWeek', 'Avg Price']
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow['Day'] = dow['DayOfWeek'].apply(lambda x: days[x])
        
        fig = px.bar(
            dow,
            x='Day',
            y='Avg Price',
            title="Average Price by Day of Week",
            color='Avg Price',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Strategic Insights")
    
    # Key findings
    st.subheader("📊 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Price Insights")
        price_change = ((filtered_df.groupby('Date')['Wholesale Price (RMB/kg)'].mean().iloc[-1] - 
                        filtered_df.groupby('Date')['Wholesale Price (RMB/kg)'].mean().iloc[0]) / 
                        filtered_df.groupby('Date')['Wholesale Price (RMB/kg)'].mean().iloc[0] * 100)
        st.metric("Price Trend", f"{price_change:+.2f}%", "Over selected period")
        
        highest_cat = filtered_df.groupby('Category Name')['Wholesale Price (RMB/kg)'].mean().idxmax()
        highest_price = filtered_df.groupby('Category Name')['Wholesale Price (RMB/kg)'].mean().max()
        st.info(f"**Highest Priced Category:** {highest_cat} (¥{highest_price:.2f}/kg)")
        
        most_volatile = filtered_df.groupby('Category Name').apply(
            lambda x: x['Wholesale Price (RMB/kg)'].std() / x['Wholesale Price (RMB/kg)'].mean()
        ).idxmax()
        st.warning(f"**Most Volatile Category:** {most_volatile}")
    
    with col2:
        st.markdown("#### Loss Insights")
        avg_loss = filtered_df['Loss Rate (%)'].mean()
        st.metric("Average Loss Rate", f"{avg_loss:.2f}%")
        
        high_loss_count = len(filtered_df[filtered_df['Loss Rate (%)'] > 15]['Item Name'].unique())
        st.error(f"**High Loss Items (>15%):** {high_loss_count} items")
        
        highest_loss_cat = filtered_df.groupby('Category Name')['Loss Rate (%)'].mean().idxmax()
        highest_loss = filtered_df.groupby('Category Name')['Loss Rate (%)'].mean().max()
        st.warning(f"**Highest Loss Category:** {highest_loss_cat} ({highest_loss:.2f}%)")
    
    st.markdown("---")
    
    # Recommendations
    st.subheader("💡 Strategic Recommendations")
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.markdown("#### 🎯 High Priority")
        st.markdown("""
        **Supply Chain Optimization**
        - Focus on {0} high-loss items (>15%)
        - Expected Impact: 5-10% margin improvement
        - Implementation: Cold chain enhancement
        """.format(high_loss_count))
    
    with rec_col2:
        st.markdown("#### ⚡ Medium Priority")
        st.markdown("""
        **Dynamic Pricing Strategy**
        - Target volatile categories
        - Expected Impact: Better margin protection
        - Implementation: Real-time pricing algorithms
        """)
    
    with rec_col3:
        st.markdown("#### 📦 Medium Priority")
        st.markdown("""
        **Inventory Optimization**
        - Seasonal demand management
        - Expected Impact: Reduced carrying costs
        - Implementation: Predictive models
        """)
    
    st.markdown("---")
    
    # Risk factors
    st.subheader("⚠️ Risk Factors")
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        price_cv = filtered_df['Wholesale Price (RMB/kg)'].std() / filtered_df['Wholesale Price (RMB/kg)'].mean()
        if price_cv > 0.5:
            st.error(f"""
            **High Price Volatility** (Severity: HIGH)
            - CV: {price_cv:.2f}
            - Mitigation: Hedging strategies, long-term contracts
            """)
    
    with risk_col2:
        if avg_loss > 10:
            st.warning(f"""
            **High Loss Rates** (Severity: MEDIUM)
            - Average: {avg_loss:.2f}%
            - Mitigation: Cold chain investment, staff training
            """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>© 2026 Vegetable Market Intelligence | Data-Driven Decision Making</p>
        <p style='font-size: 0.8em;'>Built with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)