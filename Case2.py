# Case 2: Levensverwachting over de wereld 
# In deze case wordt gekeken naar de levensverwachting over de wereld heen. Wat beïnvloed de levensverwachting nou exact. En kunnen wij met de data een goeie voorspelling maken wat de levensverwachting per land zou zijn.

# Importeer de nodige datasets
import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import os

# Set Kaggle API credentials
os.environ['KAGGLE_USERNAME'] = "samhendriks"
os.environ['KAGGLE_KEY'] = "744f6ba8e5036ce5b874b69ddf515def"

# Initialize the Kaggle API client
api = KaggleApi()
api.authenticate()

# Define the dataset identifier and download path
dataset = 'kumarajarshi/life-expectancy-who'
download_path = 'life_expectancy_data'

# Create a directory for the dataset if it does not exist
if not os.path.exists(download_path):
    os.makedirs(download_path)

# Download the dataset
api.dataset_download_files(dataset, path=download_path, unzip=True)

# Find the downloaded CSV file
csv_file = [f for f in os.listdir(download_path) if f.endswith('.csv')][0]

#Eerste kijk naar de data
# Load the dataset into a DataFrame
df = pd.read_csv(os.path.join(download_path, csv_file))

# Continent column toevoegen
def get_country_to_continent():
    # Send a GET request to the Restcountries API to fetch data about all countries
    response = requests.get('https://restcountries.com/v3.1/all')
    
    # Extract JSON data from the API response
    data = response.json()

    # Initialize an empty dictionary to store the mapping of countries to continents
    country_to_continent = {}
    
    # Loop through each country in the fetched data
    for country in data:
        # Get the common name of the country from the 'name' key
        name = country.get('name', {}).get('common', '')
        
        # Get the region (continent) of the country from the 'region' key
        region = country.get('region', '')
        
        # Add the country and continent information to the dictionary
        country_to_continent[name] = region

    # Return the dictionary containing country-to-continent mappings
    return country_to_continent

# Call the function to get the mapping of countries to continents
country_to_continent = get_country_to_continent()

# Function to fill missing continent information
def fill_missing_continent(row):
    if pd.isna(row['Continent']):
        return country_to_continent.get(row['Country'], 'Unknown')  # Use 'Unknown' for countries not in the dictionary
    return row['Continent']

# Initialize and empty column for continents
df['Continent'] = None

# Apply the function to the DataFrame
df['Continent'] = df.apply(fill_missing_continent, axis=1)

# List countries that received 'Unknown'
unknown_countries = df[df['Continent'] == 'Unknown']['Country'].value_counts()

# Define the correct continent for the specific countries
manual_corrections = {
    'Bolivia (Plurinational State of)': 'Americas',
    'Brunei Darussalam': 'Asia',
    'Venezuela (Bolivarian Republic of)': 'Americas',
    'United States of America': 'Americas',
    'United Republic of Tanzania': 'Africa',
    'United Kingdom of Great Britain and Northern Ireland': 'Europe',
    'The former Yugoslav republic of Macedonia': 'Europe',
    'Syrian Arab Republic': 'Asia',
    'Swaziland': 'Africa',
    'Sao Tome and Principe': 'Africa',
    'Russian Federation': 'Europe',
    'Republic of Moldova': 'Europe',
    'Republic of Korea': 'Asia',
    'Micronesia (Federated States of)': 'Oceania',
    'Lao People\'s Democratic Republic': 'Asia',
    'Iran (Islamic Republic of)': 'Asia',
    'Democratic Republic of the Congo': 'Africa',
    'Democratic People\'s Republic of Korea': 'Asia',
    'Congo': 'Africa',
    'Cabo Verde': 'Africa',
    'Côte d\'Ivoire': 'Africa',
    'Viet Nam': 'Asia'
}

# Update the 'Continent' column using the manual corrections
df['Continent'] = df['Country'].map(manual_corrections).fillna(df['Continent'])

# List countries that received 'Unknown'
unknown_countries = df[df['Continent'] == 'Unknown']['Country'].value_counts()

# Developing and developed changing to numerical values
# Replace developing with 0 and developed with a 1
df['Status'] = df['Status'].replace({'Developing': 0, 'Developed': 1})

# Deciding what to use and what to drop
# Create a DataFrame
df_cor = df

def correlation_matrix():
    # Select the columns you want to use
    columns_to_use = ['Life expectancy ', 'Adult Mortality', 'infant deaths', 'Alcohol', 
                      'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 
                      'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ', 
                      ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years', 
                      ' thinness 5-9 years', 'Income composition of resources', 'Schooling']

    # Create a new DataFrame with only the selected columns
    df_selected = df_cor[columns_to_use]

    # Calculate the correlation matrix
    corr_matrix = df_selected.corr()

    # Create the heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
    
        zmin=-1, zmax=1,
        hoverongaps=False,
        colorbar=dict(title='Correlation'),
        text=corr_matrix.values,  # Display the correlation values on the heatmap
        texttemplate="%{text:.2f}",  # Format the text to show two decimal places
        textfont=dict(color="white")  # Change text color for visibility
    ))

    # Update layout for better presentation
    fig.update_layout(
        title='Correlation Matrix Heatmap',
        xaxis_nticks=36,
        height=800,
        xaxis_tickangle=45
    )

    # Display the heatmap in Streamlit
    st.plotly_chart(fig)

# Based on this we are dropping Population since it doesnt seem to have a huge influence 
# Drop the population column from the dataset
df = df.drop(columns=['Population'])

# NAN-values
def missing_values():
    # Calculate the percentage of missing values per column
    missing_values_percentage = df_cor.isna().sum() / len(df_cor) * 100

    # Convert the missing values series to a DataFrame for easier plotting
    missing_values_df = missing_values_percentage.reset_index()
    missing_values_df.columns = ['Column', 'Missing Percentage']

    # Create a horizontal bar plot using Plotly
    fig = px.bar(missing_values_df, 
                  y='Column',  # Switch x and y
                  x='Missing Percentage',  # Switch x and y
                  title='Percentage of Missing Values per Column',
                  labels={'Missing Percentage': 'Percentage of Missing Values'},
                  text_auto='.2f'  # Display the percentage as text on the bars
                  )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

# Gebaseerd op deze gegevens droppen de GDP gezien de vele missende waardes en de hoge correlatie met total expenditure en Hepatitis B gezien de hoeveelheid missende waardes
# Drop the GDP and Hepatitis B column from the dataset
df = df.drop(columns=['GDP', 'Hepatitis B'])

# De rest van de missende values gaan we opvullen met median van het continent waar het in ligt
def fill_missing_with_median(df):
    # Columns to fill with median values
    cols_to_fill = [
        'Life expectancy ', 'Adult Mortality', 'Alcohol', ' BMI ',
        'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS',
        ' thinness  1-19 years', ' thinness 5-9 years',
        'Income composition of resources', 'Schooling'
    ]
    
    # Calculate medians for each continent
    medians = df.groupby('Continent')[cols_to_fill].median()
    
    # Fill missing values in specified columns with median values
    for col in cols_to_fill:
        df[col] = df.apply(lambda row: medians.loc[row['Continent'], col] if pd.isna(row[col]) else row[col], axis=1)
    
    return df

# Apply the function to your dataframe
df = fill_missing_with_median(df)

# De nan values die we nu nog overhebben droppen we

df = df.dropna()

# Data manipulatie

# Dropping target column to get features
X = df.drop(columns=['Life expectancy '])  # Features
y = df['Life expectancy ']  # Target (continuous)

# Preprocessing - handling categorical data
# Convert categorical columns like 'Continent' and others to numerical
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForestRegressor Model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Prediction
y_pred = rf_model.predict(X_test)

# Print the evaluation results
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Train the model on the entire dataset
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)

# Predict on the entire dataset
y_pred_full = rf_model.predict(X)

# Save the prediction to the full dataframe
df['Predicted Life Expectancy'] = np.round(y_pred_full, 1)

# Print evaluation results on the test set (keep the model testing as is)
y_pred_test = rf_model.predict(X_test)
print("Mean Squared Error (Test Set):", mean_squared_error(y_test, y_pred_test))
print("R-squared (Test Set):", r2_score(y_test, y_pred_test))


# Data visualization
def scatterplot_machinelearning():   
    # Create a scatter plot
    fig = go.Figure()

    # Add scatter points
    for continent in df['Continent'].unique():
        subset = df[df['Continent'] == continent]
        fig.add_trace(go.Scatter(
            x=subset['Life expectancy '],
            y=subset['Predicted Life Expectancy'],
            mode='markers',
            name=continent,
            hovertext=subset['Country'],
            marker=dict(size=10)
        ))

    # Checkbox to toggle the diagonal reference line
    show_line = st.checkbox("Show Diagonal Reference Line", value=True)

    # Add a diagonal reference line if the checkbox is checked
    if show_line:
        fig.add_shape(type='line',
                      x0=df['Life expectancy '].min(),
                      y0=df['Life expectancy '].min(),
                      x1=df['Life expectancy '].max(),
                      y1=df['Life expectancy '].max(),
                      line=dict(color='Black', width=2, dash='dash'))

    # Update layout
    fig.update_layout(
        title='Actual vs. Predicted Life Expectancy',
        xaxis_title='Life expectancy ',
        yaxis_title='Predicted Life Expectancy',
        template='seaborn'
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

def world_map():
    # Get the unique years from the dataset
    years = df['Year'].unique()
    
    # Create a slider to select the year
    selected_year = st.slider(
        "Select a year:",
        min_value=int(years.min()),
        max_value=int(years.max()),
        value=int(years.max()),  # Default to the latest year
        step=1
    )
    
    # Filter the DataFrame for the selected year
    filtered_df = df[df['Year'] == selected_year]
    
    # Create a dropdown to select the column to visualize
    columns = [
        'Country', 'Year', 'Status', 'Life expectancy ', 'Adult Mortality', 'infant deaths',
        'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ',
        'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS',
        'GDP', 'Population', ' thinness  1-19 years', ' thinness 5-9 years',
        'Income composition of resources', 'Schooling'
    ]
    
    selected_column = st.selectbox(
        "Select a column to visualize:",
        columns,
        index=columns.index('Life expectancy ')  # Default to 'Life expectancy '
    )
    
    # Create a dropdown to select color scale
    color_scales = ['Ice', 'Cividis', 'Plasma', 'Inferno', 'Magma']
    selected_color_scale = st.selectbox(
        "Select a color scale:",
        color_scales,
        index=color_scales.index('Ice')  # Default to 'Ice'
    )

    # Plot the map using Plotly (example)
    fig = px.choropleth(
        filtered_df,
        locations='Country',
        locationmode='country names',
        color=selected_column,
        color_continuous_scale=selected_color_scale,
        title=f"{selected_column} in {selected_year}",
    )
    
    st.plotly_chart(fig)



    # Filter the DataFrame for the selected year
    filtered_df = df[df['Year'] == selected_year]
    
    # Create the heatmap
    fig = px.choropleth(
        filtered_df,
        locations='Country',
        locationmode='country names',
        color='Life expectancy ',
        hover_name='Country',
        color_continuous_scale=selected_color_scale,
        title=f'Life Expectancy by Country in {selected_year}',
        template='seaborn'
    )

    # Update layout to handle countries with no data
    fig.update_traces(marker=dict(line=dict(width=0.5, color='grey')),
                      selector=dict(type='choropleth'))
    
    fig.update_layout(
        dragmode=False,  # Disable dragging
        xaxis=dict(
            fixedrange=True  # Disable zooming
        ),
        yaxis=dict(
            fixedrange=True  # Disable zooming
        )
    )


def boxplot():
    # Define a custom color palette for the continents
    color_palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']

    # Create the boxplot with hover data for country
    fig = px.box(df, x='Continent', y='Life expectancy ', 
                color='Continent', 
                title='Life Expectancy per Country by Continent', 
                color_discrete_sequence=color_palette,
                hover_data=['Country'])  # Add country to hover data

    # Display the plot in Streamlit
    st.plotly_chart(fig)

def customizable_scatter_plot():
    # Define a custom color palette for the continents
    color_palette = {
        'Africa': '#00CC96',
        'Americas': '#EF553B',
        'Asia': '#636EFA',
        'Europe': '#FFA15A',
        'Oceania': '#AB63FA'
    }

    # Get all columns except for 'Year', 'Country', and 'Status'
    columns_to_select = df.columns.difference(['Year', 'Country', 'Status', 'Life expectancy ', 'Continent'])
    
    # Sort the columns alphabetically
    sorted_columns = sorted(columns_to_select)
    
    # Create a Streamlit select box for choosing the y-axis variable
    selected_variable = st.selectbox("Select a variable for the Y-axis:", options=sorted_columns)
    
    # Calculate average values per country
    avg_df = df.groupby('Country').agg({
        'Life expectancy ': 'mean',
        selected_variable: 'mean',
        'Continent': 'first'  # Retain the continent for coloring
    }).reset_index()
    
    # Create a scatter plot using Plotly
    fig = go.Figure()
    
    # Add scatter points with continent color coding
    for continent, color in color_palette.items():
        subset = avg_df[avg_df['Continent'] == continent]
        fig.add_trace(go.Scatter(
            x=subset['Life expectancy '],
            y=subset[selected_variable],
            mode='markers',
            text=subset['Country'],  # Show country names on hover
            marker=dict(size=10, color=color),
            name=continent
        ))

    # Update layout
    fig.update_layout(
        title=f'Scatter Plot of Average Life Expectancy vs {selected_variable}',
        xaxis_title='Average Life Expectancy',
        yaxis_title=selected_variable,
        template='seaborn'
    )
    
    # Display the plot in Streamlit
    st.plotly_chart(fig)

def plot_life_expectancy_histogram():
    fig = px.histogram(df, x='Predicted Life Expectancy', template='plotly_dark', 
                       histnorm='percent', 
                       nbins=30)  # Pas het aantal bins aan als dat nodig is
    fig.update_traces(marker=dict(line=dict(width=2, color='black')))  # Optioneel, om de randen van de bins te accentueren
    fig.update_layout(bargap=0.2)  # Pas de bargap aan voor meer ruimte tussen de bins
    st.plotly_chart(fig)
    
def plot_status_violin():
    fig=px.violin(df,x='Status',y='Predicted Life Expectancy',color='Status',template='plotly_dark',box=True,title='Life expectancy Based on Countries status')
    st.plotly_chart(fig)
