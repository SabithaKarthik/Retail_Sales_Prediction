import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_option_menu import option_menu
from datetime import date
import pickle
import numpy as np

# Setting up page configuration
st.set_page_config(layout= "wide")
st.title("Retail Sales Forecast")
st.write("")

# Creating option menu in the side bar
with st.sidebar:
    selected = option_menu("Menu", ["Home","Insights","Prediction"], 
                           menu_icon= "menu-button-wide"
                          )

# READING THE CLEANED DATAFRAME
df = pd.read_csv('df_sql.csv')

# HOME PAGE
if selected == "Home":
    st.header("Project Overview")
    st.write("")
    st.write('''***Retail Sales Forecast employs advanced machine learning techniques, 
             prioritizing careful data preprocessing, feature enhancement, and comprehensive 
             algorithm assessment and selection. The streamlined Streamlit application integrates 
             Exploratory Data Analysis (EDA) to find trends, patterns, and data insights. 
             It offers users interactive tools to explore top-performing stores and departments, 
             conduct insightful feature comparisons, and obtain personalized sales forecasts. 
             With a commitment to delivering actionable insights, the project aims to optimize 
             decision-making processes within the dynamic retail landscape.***''')
    st.header("Technologies used")
    st.write("")
    st.write("***Python, Pandas, Plotly, Streamlit, Scikit-Learn, Numpy, Seaborn***")
    
# OVERVIEW PAGE
if selected == "Insights":
            # Convert Month and Year to a datetime object
            df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')

            # Group by Date and sum Weekly_Sales
            sales_over_time = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
            fig = px.line(sales_over_time,
                         title='Weekly Sales Over Time',
                         x='Date',
                         y='Weekly_Sales')
            fig.update_layout(xaxis_title='Date',
                        yaxis_title='Total Weekly Sales',
                        xaxis=dict(tickangle=45),
                        template='plotly_white')
            st.plotly_chart(fig,use_container_width=True) 
        
            # Distribution of Weekly Sales by Holiday
            fig = px.box(df,
                         title='Distribution of Weekly Sales During Holidays',
                         x='IsHoliday',
                         y='Weekly_Sales')
            st.plotly_chart(fig,use_container_width=True)

            # Aggregate the Weekly_Sales by Store and Department
            store_dept_sales = df.groupby(['Store'])['Weekly_Sales'].sum().reset_index()

            # Sort the aggregated data to identify top-performing and underperforming stores/departments
            top_performing = store_dept_sales.sort_values(by='Weekly_Sales', ascending=False).head(10)

            fig = px.bar(data_frame=top_performing,
                     x='Store',
                     y='Weekly_Sales',
                     title='Top-Performing Stores'
                    )
            fig.update_layout(xaxis_title='Store',
                        yaxis_title='Weekly Sales',
                        width=800,
                        height=600)
            st.plotly_chart(fig,use_container_width=True)

             # Compute the correlation matrix
            correlation_matrix = df[['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']].corr()
            fig = px.imshow(correlation_matrix,
                         title='Correlation Heatmap of Weekly Sales and Numerical Features',
                         text_auto=True,
                         width=800,height=600)
            st.plotly_chart(fig,use_container_width=True)

             # Aggregate the Weekly_Sales by Department
            dept_sales = df.groupby(['Dept'])['Weekly_Sales'].sum().reset_index()

            # Sort the aggregated data to identify top-performing and underperforming departments
            top_dept = dept_sales.sort_values(by='Weekly_Sales', ascending=False).head(10)

            fig = px.bar(top_dept,
                     x='Dept',
                     y='Weekly_Sales',
                     title='Top-Performing Department across Stores'
                    )
            fig.update_layout(xaxis_title='Department',
                        yaxis_title='Total Weekly Sales',
                        xaxis={'categoryorder': 'total descending'}, 
                        bargap=0.1, 
                        width=1700,
                        height=600)
            st.plotly_chart(fig,use_container_width=True)
            
            # Aggregate the Weekly_Sales by Store and Department
            store_dept_sales = df.groupby(['Store'])['Weekly_Sales'].sum().reset_index()

            # Sort the aggregated data to identify top-performing and underperforming stores/departments
            underperforming = store_dept_sales.sort_values(by='Weekly_Sales').head(10)

            fig = px.bar(underperforming,
                     x='Store',
                     y='Weekly_Sales',
                     title='Under-Performing Stores'
                    )
            fig.update_layout(xaxis_title='Store',
                        yaxis_title='Total Weekly Sales',
                        xaxis={'categoryorder': 'total descending'},
                        bargap=0.1)
            st.plotly_chart(fig,use_container_width=True)
        
# Prediction PAGE
if selected == "Prediction":
    with st.form('Prediction'):
    
        col1, col2, col3 = st.columns([0.5, 0.1, 0.5])

        with col1:

            user_date = st.date_input(label='Date', min_value=date(2010, 2, 5),
                                          max_value=date(2013, 12, 31), value=date(2010, 2, 5))

            store = st.number_input(label='Store', min_value=1, max_value=45,
                                        value=1, step=1)
        
            df_dept=df.loc[df['Store']==store]['Dept']
            dept = st.selectbox(label='Dept',
                                    options=df_dept.to_list())

            holiday = st.selectbox(label='Holiday', options=['Yes', 'No'])

            temperature = st.number_input(label='Temperature(°F)', min_value=-10.0,
                                              max_value=110.0, value=-7.29)

            fuel_price = st.number_input(label='Fuel Price', max_value=10.0,
                                             value=2.47)

            cpi = st.number_input(label='CPI', min_value=100.0,
                                      max_value=250.0, value=126.06)

        with col3:

            markdown1 = st.number_input(label='MarkDown1', value=-2781.45)

            markdown2 = st.number_input(label='MarkDown2', value=-265.76)

            markdown3 = st.number_input(label='MarkDown3', value=-179.26)

            markdown4 = st.number_input(label='MarkDown4', value=0.22)

            markdown5 = st.number_input(label='MarkDown5', value=-185.87)

            unemployment = st.number_input(label='Unemployment',
                                               max_value=20.0, value=3.68)

            button = st.form_submit_button(label='SUBMIT')

        # user entered the all input values and click the button
        if button:
            with st.spinner(text='Processing...'):

                # load the regression pickle model
                with open(r'model\model1_markdown.pkl', 'rb') as f:
                    model = pickle.load(f)

                store = df['Store'].unique().tolist()
                type = df['Type'].unique().tolist()
                size = df['Size'].unique().tolist()

                type_dict, size_dict = {}, {}

                for i in range(0, len(store)):
                    type_dict[store[i]] = type[i]
                    size_dict[store[i]] = size[i]

                holiday_dict = {'Yes': 1, 'No': 0}

                # make array for all user input values in required order for model prediction
                user_data = np.array([[user_date.day, user_date.month, user_date.year,
                                       store, dept, type_dict[store], size_dict[store],
                                       holiday_dict[holiday], temperature,
                                       fuel_price, markdown1, markdown2, markdown3,
                                       markdown4, markdown5, cpi, unemployment]])

                # model predict the selling price based on user input
                y_pred = model.predict(user_data)[0]

                # round the value with 2 decimal point (Eg: 1.35678 to 1.36)
                weekly_sales = f"{y_pred:.2f}"

                st.markdown(f'### <div class="center-text">Predicted Sales = {weekly_sales}</div>', 
                    unsafe_allow_html=True)
