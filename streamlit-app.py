import streamlit as st
import plotly.express as px
from streamlit_option_menu import option_menu
from Case2 import scatterplot_machinelearning, customizable_scatter_plot, world_map, boxplot, correlation_matrix, missing_values, plot_life_expectancy_histogram, plot_life_expectancy_histogram, plot_status_violin

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&display=swap');

    /* Background of the main app container */
    .reportview-container {
        background-color: #f4f4f9;
        padding: 10px;
    }

    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        padding: 10px;
        width: 50px;  /* Pas de breedte hier aan */
    }

    /* Main title styling */
    h1 {
        font-family: 'Lato', sans-serif;
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        font-size: 42px;
        margin-bottom: 25px;
    }

    /* Section Header styling */
    h2, h3 {
        font-family: 'Lato', sans-serif;
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 10px;
        font-size: 30px;
    }

    /* Styling for paragraph and content */
    p, .element-container {
        font-family: 'Lato', sans-serif;
        font-size: 18px;
        line-height: 1.6;
        color: #34495e;
        padding: 10px;
    }

    /* Styling the box and correlation matrices headers */
    .header-style {
        font-size: 24px;
        color: #2c3e50;
        font-weight: 600;
        margin-top: 40px;
        margin-bottom: 20px;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        padding: 20px;
        font-size: 14px;
        color: #7f8c8d;
        border-top: 1px solid #dcdcdc;
        margin-top: 50px;
    }

    /* Hover effect for interactive elements */
    .element-container:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: 0.3s ease;
    }

    /* Add padding for better text readability */
    .element-container {
        padding: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar with improved navigation using option_menu
with st.sidebar:
    page = option_menu(
        menu_title="Navigation",
        options=["Life Expectancy", "Worldmap", "Prediction"],
        icons=["person", "globe", "lightning"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f0f2f6"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#fafafa"},
            "nav-link-selected": {"background-color": "#2c3e50", "color": "white"},
        }
    )
    
# Page 1: Life Expectancy Section
if page == "Life Expectancy":
    st.markdown("<h1>Life Expectancy Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("""<div class='element-container'>
                   Welcome to the Life Expectancy Dashboard. This dashboard provides insights into global life expectancy trends and their determinants.
                   </div>""", unsafe_allow_html=True)

    # Life Expectancy Histogram
    st.markdown('<h2 class="header-style">Life Expectancy Histogram</h2>', unsafe_allow_html=True)
    st.write("""This histogram displays the distribution of life expectancy across countries. It provides insight into how life expectancy varies globally.""")
    plot_life_expectancy_histogram()

    st.markdown('<h2 class="header-style">Status Violin Plot</h2>', unsafe_allow_html=True)
    st.write("""A violin plot effectively visualizes the distribution of data, specifically showing life expectancy in relation to status. Its shape highlights how data is spread across different predictions.""")
    plot_status_violin()

    # Side-by-side layout
    col1, col2 = st.columns(2)

    # Life expectancy box plot
    with col1:
        st.markdown('<h2 class="header-style">Life Expectancy Boxplot by Continent</h2>', unsafe_allow_html=True)
        st.write("""The boxplot illustrates the variations in life expectancy across continents. European countries generally show higher life expectancy with less variability, while African nations exhibit the lowest and most varied values. The spread across Asia, America, and Oceania also highlights regional disparities.""")
        boxplot()

    # Missing values plot
    with col2:
        st.markdown('<h2 class="header-style">Percentage of Missing Data</h2>', unsafe_allow_html=True)
        st.write("""This plot displays the percentage of missing data for each column. Columns like Hepatitis B and GDP have significant gaps, leading us to exclude them. Missing values in other columns were filled using the median value for each continent.""")
        missing_values()


    # Customizable Scatter Plot
    st.markdown('<h2 class="header-style">Customizable Scatter Plot</h2>', unsafe_allow_html=True)
    st.write("""Adjust this scatterplot using the dropdown menu to explore how life expectancy relates to various factors, such as income, schooling, and healthcare expenditure.""")
    customizable_scatter_plot()

    # Footer
    st.markdown('<div class="footer">© 2024 Sam Hendriks | Data Source: World Health Organization</div>', unsafe_allow_html=True)



# Page 2: Life Expectancy Section
if page == "Worldmap":
    st.markdown("<h1>Worldmap Life Expectancy</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='element-container'>
    On this world map, you can explore life expectancy across different countries. 
    Discover the variations in life expectancy and the factors influencing them. 
    Use the map to gain deeper insights into global health trends.
    </div>
    """, unsafe_allow_html=True)
    
    # Global Life Expectancy Map
    world_map()


 # Footer
    st.markdown('<div class="footer">© 2024 Sam Hendriks | Data Source: World Health Organization</div>', unsafe_allow_html=True)


# Page 3: Life Expectancy Section
if page == "Prediction":
    # Correlation matrix
    st.markdown("<h1>Life Expectancy Predictions</h1>", unsafe_allow_html=True)
    st.write("""The correlation matrix reveals the relationships between life expectancy and various factors. Strong positive correlations were found between schooling (0.75), income composition (0.72), and life expectancy. On the contrary, high adult mortality (-0.70) and HIV/AIDS prevalence (-0.56) show strong negative correlations with life expectancy.""")
    correlation_matrix()
    
    st.markdown('<h2 class="header-style">Machine Learning Scatter Plot</h2>', unsafe_allow_html=True)
    st.write("""This scatterplot compares predicted life expectancy values to actual values using a machine learning model. The closer the points are to the diagonal line, the more accurate the model. The model performs reasonably well, though there are a few outliers.""")
    scatterplot_machinelearning()

   # Footer
    st.markdown('<div class="footer">© 2024 Sam Hendriks | Data Source: World Health Organization</div>', unsafe_allow_html=True)

