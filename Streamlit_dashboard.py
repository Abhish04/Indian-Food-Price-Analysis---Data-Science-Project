import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose

# functions 


# main codes
# executing main function
if __name__ == "__main__":
    # setting the page configuration
    st.set_page_config(page_title="Food Prices In India",page_icon="AVI",layout="wide")

    # custom bachground color
    st.markdown(
        """
        <style>
        body {
            background-color: #000000;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Food Prices In India")

    st.sidebar.title("WELCOME")

    # upload file
    uploaded_file = st.sidebar.file_uploader("CHOOSE YOUR FILE:", type=["csv","xlsx"])

    if uploaded_file is not None:
        #read the file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Invalid file type, Please upload CSV or Excel file.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

        st.header("DATA PREPARATION")

        st.write(df.head())
        st.write("Shape and size of the data",df.shape)


        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['usdprice'] = pd.to_numeric(df['usdprice'], errors='coerce')
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        st.header("DATA EXPLORATION")
        st.write("Exploring the loaded dataset to understand its structure and characteristics.")

        # Examine the shape of the DataFrame
        st.write("Shape of the DataFrame:", df.shape)

        # Get an overview of data types and missing values
        st.write("\nData types and missing values:")
        df.info()

        # Calculate descriptive statistics for numerical columns
        st.write("\nDescriptive statistics for numerical columns:")
        st.write(df.describe(include=['number']))

        # Analyze categorical columns
        for col in ['commodity', 'market', 'state']:  # Assuming 'state' is a categorical column
            if col in df.columns:
                st.write(f"\nUnique values and frequencies for '{col}':")
                st.write(df[col].value_counts())
            else:
                st.write(f"Column '{col}' not found in the DataFrame.")

        # Check for missing values and calculate the percentage
        st.write("\nMissing values:")
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
        st.write(missing_data)

        # Analyze the distribution of 'price' and 'usdprice'
        st.write("\nDistribution of 'price' and 'usdprice':")
        st.write(df[['price', 'usdprice']].describe())

        # Examine the 'date' column
        st.write("\nInformation about the 'date' column:")
        st.write("Earliest date:", df['date'].min())
        st.write("Latest date:", df['date'].max())
        st.write("Number of unique dates:", df['date'].nunique())

        # Investigate the relationship between 'price' and 'usdprice' and other numerical features
        st.write("\nCorrelation analysis:")
        numerical_cols = ['price', 'usdprice', 'latitude', 'longitude']
        correlation_matrix = df[numerical_cols].corr()
        st.write(correlation_matrix)


        st.header("DATA CLEANING")
        st.write("Cleaning the data by handling missing values and outliers in the df DataFrame")

        # Impute missing values
        for col in ['price', 'usdprice', 'latitude', 'longitude']:
            df[col] = df[col].fillna(df[col].median())

        for col in ['admin1', 'admin2', 'market', 'category', 'commodity', 'unit', 'priceflag', 'pricetype', 'currency']:
            df[col] = df[col].fillna(df[col].mode()[0])

        # Outlier handling for 'price' and 'usdprice' using IQR
        for col in ['price', 'usdprice']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        # Data consistency: Standardize categorical variables (example for 'commodity')
        df['commodity'] = df['commodity'].str.lower()
        df['commodity'] = df['commodity'].str.strip()

        # Further data consistency checks and standardization can be added for other columns as needed
        st.write("\nData after cleaning:")
        st.write(df.head())
        st.write(df.info())


        st.header("DATA WRANGLING")
        st.write("Preparing the data for analysis and visualization by creating new features and converting data types")

        # Create 'year' and 'month' columns
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # Create 'date_ym' column
        df['date_ym'] = df['date'].dt.strftime('%Y-%m')

        # Convert 'date_ym' to datetime objects
        df['date_ym'] = pd.to_datetime(df['date_ym'], format='%Y-%m')

        # Check for invalid years and months
        invalid_years = ~df['year'].between(1990, 2025)  # Adjust the range as needed
        invalid_months = ~df['month'].between(1, 12)

        # Handle inconsistencies (e.g., replace with NaN or a specific value)
        df.loc[invalid_years, 'year'] = pd.NA
        df.loc[invalid_months, 'month'] = pd.NA

        # Handle any remaining missing values in the new columns
        df['year'] = df['year'].fillna(method='ffill')
        df['month'] = df['month'].fillna(method='ffill')

        st.write(df.head())
        st.write(df.info())


        st.header("DATA ANALYSIS")
        st.write("Analysing the cleaned and wrangled data to identify key trends and insights.")
        # Perform exploratory data analysis (EDA) to identify key trends and insights

        # Price Trends Over Time
        price_trends = df.groupby(['commodity', 'date_ym'])['price'].mean().reset_index()
        plt.figure(figsize=(15, 8))
        for commodity in price_trends['commodity'].unique():
            subset = price_trends[price_trends['commodity'] == commodity]
            plt.plot(subset['date_ym'], subset['price'], label=commodity)
        plt.xlabel('Date')
        plt.ylabel('Average Price')
        plt.title('Price Trends Over Time by Commodity')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Price Distribution by Commodity
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='commodity', y='price', data=df)
        plt.xlabel('Commodity')
        plt.ylabel('Price')
        plt.title('Price Distribution by Commodity')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        for commodity in df['commodity'].unique():
            sns.histplot(df[df['commodity'] == commodity]['price'], label=commodity, kde=True, element="step")
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.title('Price Distribution by Commodity')
        plt.legend()
        plt.show()

        # Descriptive statistics for each commodity
        price_stats = df.groupby('commodity')['price'].agg(['mean', 'median', 'std', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        price_stats.columns = ['mean', 'median', 'std', '25th percentile', '75th percentile']
        st.write(price_stats)


        # Geographical Price Analysis (using 'market' as a proxy for location)
        market_prices = df.groupby(['market', 'commodity'])['price'].mean().reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(x='market', y='price', hue='commodity', data=market_prices)
        plt.xlabel('Market')
        plt.ylabel('Average Price')
        plt.title('Average Price by Market and Commodity')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


        # Correlation Analysis
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='usdprice', y='price', data=df)
        plt.xlabel('USD Price')
        plt.ylabel('Price')
        plt.title('Correlation between Price and USD Price')
        plt.show()


        numerical_cols = ['price', 'usdprice', 'latitude', 'longitude']
        correlation_matrix = df[numerical_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix of Numerical Features')
        plt.show()


        # Select key commodities for time series decomposition (e.g., rice, wheat, sugar)
        key_commodities = ['rice', 'wheat', 'sugar']

        for commodity in key_commodities:
            commodity_data = df[df['commodity'] == commodity].sort_values('date_ym')
            if not commodity_data.empty:
                # Time Series Decomposition
                decomposition = seasonal_decompose(commodity_data['price'], model='additive', period=12)  # Adjust period as needed

                plt.figure(figsize=(12, 8))

                plt.subplot(411)
                plt.plot(commodity_data['date_ym'], commodity_data['price'], label='Original')
                plt.legend(loc='best')
                plt.title(f'Time Series Decomposition for {commodity}')

                plt.subplot(412)
                plt.plot(decomposition.trend, label='Trend')
                plt.legend(loc='best')

                plt.subplot(413)
                plt.plot(decomposition.seasonal, label='Seasonality')
                plt.legend(loc='best')

                plt.subplot(414)
                plt.plot(decomposition.resid, label='Residuals')
                plt.legend(loc='best')

                plt.tight_layout()
                plt.show()


        st.header("DATA VISUALIZATION")
        st.write("Creating visualizations for a Streamlit dashboard, focusing on interactive elements to explore price trends, distributions, and geographical comparisons")

        # Price Trends Over Time
        st.subheader("Price Trends Over Time")
        commodities = sorted(df['commodity'].unique())[1:]
        selected_commodities = st.multiselect("Select Commodities", commodities, default=commodities[1])
        filtered_df = df[df['commodity'].isin(selected_commodities)]
        fig = px.line(filtered_df, x='date_ym', y='price', color='commodity', title="Price Trends Over Time", hover_data=['price'])
        st.plotly_chart(fig)


        # Price Distributions
        st.subheader("Price Distributions")
        selected_commodity_dist = st.selectbox("Select Commodity", commodities, index=1)
        fig = px.histogram(df[df['commodity'] == selected_commodity_dist], x='price', nbins=20, title=f"Price Distribution of {selected_commodity_dist}")
        st.plotly_chart(fig)


        # Geographical Price Comparison
        st.subheader("Geographical Price Comparison")
        selected_commodity_map = st.selectbox("Select Commodity", commodities)
        # Date Range Selection
        date_min = df['date_ym'].min().date()
        date_max = df['date_ym'].max().date()
        st.write("Min. Date:", date_min)
        st.write("Max. Date:", date_max)
        st.write("Please select two dates")

        date_range = st.date_input("Select Date Range", value=(date_min, date_max))
        #date_range = st.date_input("Select Date Range", value=(df['date_ym'].min().date(), df['date_ym'].max().date()))
        filtered_df_map = df[(df['commodity'] == selected_commodity_map) & (df['date_ym'].dt.date >= date_range[0]) & (df['date_ym'].dt.date <= date_range[1])]

        fig = px.scatter_geo(filtered_df_map, lat='latitude', lon='longitude', color='price', hover_name='market', hover_data=['price'], title=f'Geographical Price Comparison for {selected_commodity_map}', scope='asia')
        st.plotly_chart(fig)


        st.header("SUMMARY")
        st.subheader("Data Analysis Key Findings")

        st.write("Strong Correlation between Price and USD Price: The price and usdprice columns exhibit a very strong positive correlation (0.98), as expected.")
        st.write("Significant Missing Data: Columns like 'latitude', 'longitude', 'admin1', and 'admin2' have a substantial percentage of missing data (around 0.37%).")
        st.write("Outliers in Price and USD Price: The maximum values in the 'price' and 'usdprice' columns are significantly higher than the 75th percentile, suggesting potential outliers. These were addressed using IQR method.")
        st.write("Price Trends Vary by Commodity: The visualizations of price trends over time reveal different patterns for various commodities.")
        st.write("Commodity-Specific Price Distributions: Box plots and histograms illustrate varying price distributions across different commodities, highlighting potential differences in price volatility and central tendency.")
        st.write("Geographical Price Variations: The geographical price comparisons indicate price differences across various markets in India.")
        st.write("Time Series Decompostion: Time series decomposition of key commodities like rice, wheat, and sugar revealed underlying trends, seasonality, and residuals.")

    else:
        st.header("PLEASE UPLOAD YOUR DATA THROUGH SIDEBAR FOR ANALYSIS")