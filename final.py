import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="BMW Car Sales Analysis Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Set background image
def set_bg_hack():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://wallpapers.com/images/hd/hood-and-headlight-bmw-laptop-gb8fftjuiv6jkdmj.jpg");
             background-size: cover;
             background-attachment: fixed;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_bg_hack()
# Custom dark theme for plots
dark_theme = {
    'dark': True,
    'detail': '#007439',
    'primary': '#00FF41',
    'secondary': '#6D326D',
}

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('BMW_Car_Sales_Classification.csv')
        return df
    except FileNotFoundError:
        st.error("BMW_Car_Sales_Classification.csv file not found in data directory")
        st.stop()

df = load_data()

# Sidebar with semi-transparent background
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: rgba(0,0,0,0.8);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("BMW Sales Dashboard")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/BMW.svg/1200px-BMW.svg.png", 
                 width=100)
st.sidebar.markdown("Analyze BMW car sales data with interactive visualizations.")

# Title and description
st.title("🚗 BMW Car Sales Analysis Dashboard")
st.markdown("### Comprehensive analysis of 50,000 BMW car sales records (2010-2024)")

# # Sidebar filters
# st.sidebar.header("Filters")

# # Year filter
# year_range = st.sidebar.slider(
#     "Year Range",
#     min_value=(df['Year'].min()),
#     max_value=(df['Year'].max()),
#     value=(int(df['Year'].min()), int(df['Year'].max())),
#     step=1
# )

# # Model filter
# models = st.sidebar.multiselect(
#     "Select Models",
#     options=sorted(df['Model'].unique()),
#     default=sorted(df['Model'].unique())
# )

# # Region filter
# regions = st.sidebar.multiselect(
#     "Select Regions",
#     options=sorted(df['Region'].unique()),
#     default=sorted(df['Region'].unique())
# )

# # Sales Classification filter
# sales_class = st.sidebar.multiselect(
#     "Sales Classification",
#     options=sorted(df['Sales_Classification'].unique()),
#     default=sorted(df['Sales_Classification'].unique())
# )

# # Filter data
# filtered_df = df[
#     (df['Year'] >= year_range[0]) & 
#     (df['Year'] <= year_range[1]) &
#     (df['Model'].isin(models)) &
#     (df['Region'].isin(regions)) &
#     (df['Sales_Classification'].isin(sales_class))
# ]

# Main dashboard tabs - Reorganized order
st.sidebar.header("Filters")
selected_models = st.sidebar.multiselect("Select Models", df['Model'].unique(), default=df['Model'].unique())
selected_years = st.sidebar.slider("Select Year Range", 
                                  min_value=int(df['Year'].min()), 
                                  max_value=int(df['Year'].max()),
                                  value=(int(df['Year'].min()), int(df['Year'].max())))
selected_regions = st.sidebar.multiselect("Select Regions", df['Region'].unique(), default=df['Region'].unique())

# Apply filters
filtered_df = df[
    (df['Model'].isin(selected_models)) &
    (df['Year'].between(selected_years[0], selected_years[1])) &
    (df['Region'].isin(selected_regions))
]

# Show filtered data
st.sidebar.header("Filtered Data Stats")
st.sidebar.write(f"Records: {len(filtered_df)}")
st.sidebar.write(f"Average Price: ${filtered_df['Price_USD'].mean():,.2f}")
st.sidebar.write(f"Total Sales: {filtered_df['Sales_Volume'].sum():,}")

# Data download
st.sidebar.header("Data Export")
st.sidebar.download_button(
    label="Download Filtered Data as CSV",
    data=filtered_df.to_csv().encode('utf-8'),
    file_name='filtered_bmw_sales.csv',
    mime='text/csv'
)

# About section
st.sidebar.header("About")
st.sidebar.info("""
This dashboard provides interactive visualizations of BMW car sales data. 
Use the filters to explore specific models, years, and regions.
""")
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Overview", 
    "📊 Sales Trends", 
    "🌍 Regional Analysis", 
    "🚙 Model Performance", 
    "💰 Pricing Insights",
    "🛠 Data Quality",
    "💡 Insights & Recommendations"
])

with tab1:
    st.header("Dataset Overview")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Records", f"{len(filtered_df):,}")
    
    with col2:
        st.metric("Total Sales Volume", f"{filtered_df['Sales_Volume'].sum():,}")
    
    with col3:
        st.metric("Average Price", f"${filtered_df['Price_USD'].mean():,.0f}")
    
    with col4:
        st.metric("Average Mileage", f"{filtered_df['Mileage_KM'].mean():,.0f} km")
    
    with col5:
        st.metric("Models Available", len(filtered_df['Model'].unique()))
    
    # Data sample
    st.subheader("Data Sample")
    st.dataframe(filtered_df.head(10), use_container_width=True)
    
    # Basic statistics
    st.subheader("Statistical Summary")
    st.dataframe(filtered_df.describe(), use_container_width=True)

with tab2:
    st.header("Sales Trends Analysis")
    
    # Sales trends over time
    st.subheader("Sales Trends Over Time")
    yearly_sales = filtered_df.groupby('Year').agg({
        'Sales_Volume': 'sum',
        'Price_USD': 'mean'
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=yearly_sales['Year'], y=yearly_sales['Sales_Volume'], 
                  name="Sales Volume", line=dict(color="blue")),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=yearly_sales['Year'], y=yearly_sales['Price_USD'], 
                  name="Average Price", line=dict(color="red")),
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Sales Volume", secondary_y=False)
    fig.update_yaxes(title_text="Average Price ($)", secondary_y=True)
    fig.update_layout(title_text="Sales Volume and Average Price Trends")
    st.plotly_chart(fig, use_container_width=True)
    
    # Sales Classification Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales Classification Distribution")
        sales_dist = filtered_df['Sales_Classification'].value_counts()
        fig = px.pie(
            values=sales_dist.values,
            names=sales_dist.index,
            title="Sales Classification Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sales Volume by Classification")
        sales_vol = filtered_df.groupby('Sales_Classification')['Sales_Volume'].sum().reset_index()
        fig = px.bar(
            sales_vol,
            x='Sales_Classification',
            y='Sales_Volume',
            title="Total Sales Volume by Classification",
            color='Sales_Classification'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Transmission and Fuel Type Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transmission Type Distribution")
        trans_dist = filtered_df['Transmission'].value_counts()
        fig = px.pie(values=trans_dist.values, names=trans_dist.index)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Fuel Type Distribution")
        fuel_dist = filtered_df['Fuel_Type'].value_counts()
        fig = px.pie(values=fuel_dist.values, names=fuel_dist.index)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Regional Analysis")
    
    # Sales by region
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales Volume by Region")
        regional_sales = filtered_df.groupby('Region')['Sales_Volume'].sum().sort_values(ascending=False)
        fig = px.bar(
            x=regional_sales.index,
            y=regional_sales.values,
            title="Total Sales Volume by Region"
        )
        fig.update_xaxes(title_text="Region")
        fig.update_yaxes(title_text="Sales Volume")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Average Price by Region")
        regional_price = filtered_df.groupby('Region')['Price_USD'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=regional_price.index,
            y=regional_price.values,
            title="Average Price by Region",
            color=regional_price.values,
            color_continuous_scale='viridis'
        )
        fig.update_xaxes(title_text="Region")
        fig.update_yaxes(title_text="Average Price ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Regional preferences
    st.subheader("Regional Model Preferences")
    region_model = filtered_df.groupby(['Region', 'Model'])['Sales_Volume'].sum().reset_index()
    fig = px.sunburst(
        region_model,
        path=['Region', 'Model'],
        values='Sales_Volume',
        title="Regional Model Preferences (Hierarchical View)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Color preferences by region
    st.subheader("Color Preferences by Region")
    color_region = pd.crosstab(filtered_df['Region'], filtered_df['Color'], normalize='index') * 100
    fig = px.imshow(
        color_region.values,
        x=color_region.columns,
        y=color_region.index,
        color_continuous_scale='viridis',
        title="Color Preferences by Region (%)"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Model Performance")
    
    # Model performance metrics
    model_stats = filtered_df.groupby('Model').agg({
        'Sales_Volume': ['sum', 'mean'],
        'Price_USD': ['mean', 'median'],
        'Mileage_KM': 'mean',
        'Engine_Size_L': 'mean'
    }).round(2)
    
    model_stats.columns = ['Total_Sales', 'Avg_Sales_Volume', 'Mean_Price', 'Median_Price', 'Avg_Mileage', 'Avg_Engine_Size']
    model_stats = model_stats.reset_index()
    
    st.subheader("Model Performance Metrics")
    st.dataframe(model_stats, use_container_width=True)
    
    # Model comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales Performance vs Price")
        fig = px.scatter(
            model_stats,
            x='Mean_Price',
            y='Total_Sales',
            size='Avg_Engine_Size',
            hover_name='Model',
            title="Sales Performance vs Price (Size = Engine Size)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Sales Distribution")
        model_sales = filtered_df.groupby('Model')['Sales_Volume'].sum().sort_values(ascending=False)
        fig = px.pie(
            values=model_sales.values,
            names=model_sales.index,
            title="Market Share by Model"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed model analysis
    st.subheader("Model-wise Sales Classification")
    model_class = pd.crosstab(filtered_df['Model'], filtered_df['Sales_Classification'], normalize='index') * 100
    fig = px.imshow(
        model_class.values,
        x=model_class.columns,
        y=model_class.index,
        color_continuous_scale='RdYlBu_r',
        title="Sales Classification by Model (%)"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Pricing Insights")
    
    # Price distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution")
        fig = px.histogram(
            filtered_df,
            x='Price_USD',
            nbins=30,
            title="Price Distribution",
            color_discrete_sequence=['skyblue']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Mileage Distribution")
        fig = px.histogram(
            filtered_df,
            x='Mileage_KM',
            nbins=30,
            title="Mileage Distribution",
            color_discrete_sequence=['lightgreen']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Price vs Mileage relationship
    st.subheader("Price vs Mileage Relationship")
    fig = px.scatter(
        filtered_df,
        x='Mileage_KM',
        y='Price_USD',
        color='Sales_Classification',
        size='Sales_Volume',
        hover_data=['Model', 'Year', 'Region'],
        title="Price vs Mileage (Size = Sales Volume)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Price by various factors
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Price by Model")
        price_by_model = filtered_df.groupby('Model')['Price_USD'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=price_by_model.index,
            y=price_by_model.values,
            title="Average Price by Model"
        )
        fig.update_xaxes(title_text="Model")
        fig.update_yaxes(title_text="Average Price ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Price by Engine Size")
        fig = px.scatter(
            filtered_df,
            x='Engine_Size_L',
            y='Price_USD',
            color='Model',
            title="Price vs Engine Size"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.header("Data Quality & Information")
    
    # Data quality checks
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Missing Values Check")
        missing_data = filtered_df.isnull().sum()
        if missing_data.sum() == 0:
            st.success("✅ No missing values found in the dataset!")
        else:
            st.warning("⚠️ Missing values detected:")
            st.write(missing_data[missing_data > 0])
    
    with col2:
        st.subheader("Data Types")
        data_types = pd.DataFrame({
            'Column': filtered_df.dtypes.index,
            'Data Type': filtered_df.dtypes.values
        })
        st.dataframe(data_types, use_container_width=True)
    
    # Unique values count
    st.subheader("Unique Values Count")
    unique_counts = pd.DataFrame({
        'Column': filtered_df.columns,
        'Unique Values': [filtered_df[col].nunique() for col in filtered_df.columns],
        'Sample Values': [str(list(filtered_df[col].unique()[:5])) for col in filtered_df.columns]
    })
    st.dataframe(unique_counts, use_container_width=True)
    
    # Correlation matrix
    st.subheader("Correlation Matrix")
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    corr_matrix = filtered_df[numeric_cols].corr()
    fig = px.imshow(
        corr_matrix,
        title="Correlation Matrix of Numeric Variables",
        color_continuous_scale='RdBu',
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Export functionality
    st.subheader("Export Data")
    if st.button("Download Filtered Data as CSV"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"bmw_sales_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with tab7:
    st.header("💡 Key Insights & Strategic Recommendations")
    
    # Calculate key insights
    best_model = filtered_df.groupby('Model')['Sales_Volume'].sum().idxmax()
    best_model_sales = filtered_df.groupby('Model')['Sales_Volume'].sum().max()
    
    highest_price_model = filtered_df.groupby('Model')['Price_USD'].mean().idxmax()
    highest_price = filtered_df.groupby('Model')['Price_USD'].mean().max()
    
    best_region = filtered_df.groupby('Region')['Sales_Volume'].sum().idxmax()
    best_region_sales = filtered_df.groupby('Region')['Sales_Volume'].sum().max()
    
    most_popular_color = filtered_df['Color'].value_counts().idxmax()
    most_popular_transmission = filtered_df['Transmission'].value_counts().idxmax()
    most_popular_fuel = filtered_df['Fuel_Type'].value_counts().idxmax()
    
    # High performers vs Low performers
    high_sales_models = filtered_df.groupby('Model')['Sales_Volume'].sum().nlargest(3).index.tolist()
    low_sales_models = filtered_df.groupby('Model')['Sales_Volume'].sum().nsmallest(3).index.tolist()
    
    # Price vs Sales correlation
    price_sales_corr = filtered_df['Price_USD'].corr(filtered_df['Sales_Volume'])
    
    # Year-over-year growth
    yearly_growth = filtered_df.groupby('Year')['Sales_Volume'].sum().pct_change().mean() * 100
    
    # Sales classification analysis
    high_class_percentage = (filtered_df['Sales_Classification'] == 'High').mean() * 100
    
    st.subheader("🔍 Key Business Insights")
    
    # Insights in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Performance Insights")
        st.info(f"""
        **Top Performing Model:** {best_model}
        - Total Sales Volume: {best_model_sales:,} units
        
        **Premium Model:** {highest_price_model}  
        - Average Price: ${highest_price:,.0f}
        
        **Leading Market:** {best_region}
        - Sales Volume: {best_region_sales:,} units
        
        **Market Growth Rate:** {yearly_growth:.1f}% YoY average
        """)
        
        st.markdown("### 🎯 Customer Preferences")
        st.success(f"""
        **Most Popular Specifications:**
        - Color: {most_popular_color}
        - Transmission: {most_popular_transmission}  
        - Fuel Type: {most_popular_fuel}
        
        **High Sales Classification:** {high_class_percentage:.1f}% of total sales
        """)
    
    with col2:
        st.markdown("### 📊 Market Analysis")
        st.warning(f"""
        **Price-Sales Correlation:** {price_sales_corr:.3f}
        {'Strong negative correlation - Higher prices tend to have lower sales' if price_sales_corr < -0.3 else 'Moderate correlation between price and sales' if abs(price_sales_corr) < 0.3 else 'Strong positive correlation - Premium pricing strategy works'}
        
        **Top 3 Performers:**
        {', '.join(high_sales_models)}
        
        **Underperformers:**
        {', '.join(low_sales_models)}
        """)
    
    st.subheader("🚀 Strategic Recommendations")
    
    # Strategic recommendations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 💼 Sales Strategy")
        st.markdown(f"""
        **1. Focus on Winners**
        - Increase marketing spend on **{best_model}** model
        - Expand inventory in **{best_region}** region
        - Leverage success factors of top performers
        
        **2. Premium Positioning** 
        - Continue premium strategy for **{highest_price_model}**
        - Develop luxury variants for high-performing models
        """)
    
    with col2:
        st.markdown("### 🎨 Product Strategy")
        st.markdown(f"""
        **1. Optimize Offerings**
        - Prioritize **{most_popular_color}** color in production
        - Focus on **{most_popular_transmission}** transmission
        - Expand **{most_popular_fuel}** fuel type options
        
        **2. Revitalize Underperformers**
        - Review pricing strategy for: {', '.join(low_sales_models[:2])}
        - Consider feature upgrades or discontinuation
        """)
    
    with col3:
        st.markdown("### 🌍 Market Expansion")
        st.markdown(f"""
        **1. Regional Focus**
        - Replicate **{best_region}** success in other regions
        - Analyze regional preferences for targeted marketing
        
        **2. Growth Opportunities**
        - Investigate markets with lower penetration
        - Develop region-specific model variants
        - Consider partnerships in emerging markets
        """)
    
    # Action Items
    st.subheader("✅ Immediate Action Items")
    
    action_items = [
        f"📊 Conduct deep-dive analysis on {best_model} success factors",
        f"💰 Review pricing strategy for underperforming models: {', '.join(low_sales_models[:2])}",
        f"🌟 Launch targeted campaign in {best_region} to maintain market leadership",
        f"🔍 Investigate why {most_popular_color} cars are most preferred across markets",
        "📈 Develop quarterly performance tracking dashboard",
        "🎯 Create customer segmentation strategy based on regional preferences"
    ]
    
    for i, item in enumerate(action_items, 1):
        st.markdown(f"**{i}.** {item}")
    
    # ROI Projections
    st.subheader("💡 Projected Impact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Potential Revenue Increase", 
            "15-25%",
            "By focusing on top performers"
        )
    
    with col2:
        st.metric(
            "Market Share Growth",
            "3-5%", 
            "Through regional expansion"
        )
    
    with col3:
        st.metric(
            "Cost Optimization",
            "10-15%",
            "By discontinuing underperformers"
        )
    
    # Risk Assessment
    st.subheader("⚠️ Risk Assessment & Mitigation")
    
    st.error("""
    **Key Risks:**
    - Over-dependence on top-performing models
    - Regional market concentration risk
    - Price sensitivity in premium segments
    
    **Mitigation Strategies:**
    - Diversify model portfolio
    - Develop contingency plans for market downturns  
    - Monitor competitor pricing strategies
    """)
# Footer
st.markdown("---")
st.markdown("### About this Dashboard")
st.markdown("""
This dashboard provides comprehensive analysis of BMW car sales data containing 50,000 records from 2010-2024.
The analysis includes sales patterns, pricing trends, regional preferences, model comparisons, and strategic insights.
Use the sidebar filters to explore specific segments of the data.
""")
# C:\Users\Lenovo\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe -m streamlit run final.py