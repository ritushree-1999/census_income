import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


@st.cache()
def load_data():
	# Load the Adult Income dataset into DataFrame.

	df = pd.read_csv('https://student-datasets-bucket.s3.ap-south-1.amazonaws.com/whitehat-ds-datasets/adult.csv', header=None)
	df.head()

	# Rename the column names in the DataFrame. 

	# Create the list
	column_name =['age', 'workclass', 'fnlwgt', 'education', 'education-years', 'marital-status', 'occupation', 'relationship', 'race', 'gender','capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

	# Rename the columns using 'rename()'
	for i in range(df.shape[1]):
	  df.rename(columns={i:column_name[i]},inplace=True)

	# Print the first five rows of the DataFrame
	df.head()

	# Replace the invalid values ' ?' with 'np.nan'.

	df['native-country'] = df['native-country'].replace(' ?',np.nan)
	df['workclass'] = df['workclass'].replace(' ?',np.nan)
	df['occupation'] = df['occupation'].replace(' ?',np.nan)

	# Delete the rows with invalid values and the column not required 

	# Delete the rows with the 'dropna()' function
	df.dropna(inplace=True)

	# Delete the column with the 'drop()' function
	df.drop(columns='fnlwgt',axis=1,inplace=True)

	return df

census_df = load_data()

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Adult Income Predictor")
st.sidebar.title("Census Data Analysis")

if st.sidebar.checkbox("Show raw data"):
  st.subheader("Census Data Set")
  st.dataframe(census_df)
  st.write("Number of rows",census_df.shape[0])
  st.write("Number of columns",census_df.shape[1])

  st.sidebar.subheader("Visualisation Selector")

# Add a multiselect in the sidebar with label 'Select the Charts/Plots:'
# Store the current value of this widget in a variable 'plot_list'.
plot_list=st.sidebar.multiselect("Select the Charts/Plots:",
                                  ("Pie Chart","Box Plot","Count Plot"))


features_list = st.sidebar.multiselect("Select the x-axis values:", 
                                    ('income','gender','workclass','hours-per-week'))

# Create box plot using the 'seaborn' module and the 'st.pyplot()' function.
if 'Box Plot' in plot_list:
    st.subheader("Box Plot for the Hour Per Week")
    columns = st.sidebar.selectbox("Select the column to create its box plot",
                                  ('income','gender','hours-per-week'))

    st.subheader(f"Box Plots {columns}")
    plt.figure(figsize = (12, 2))
    plt.title(f"Distribution of hour pre week and {columns} features for different groups")
    sns.boxplot(x=census_df['hours-per-week'],y=census_df[columns])
    st.pyplot()

# Create count plot using the 'seaborn' module and the 'st.pyplot()' function.
if 'Count Plot' in plot_list:
    st.subheader("Count plot number of records for unique workclass feature values")
    sns.countplot(x = 'workclass',hue = 'income', data = census_df)
    st.pyplot()

# Create a pie chart using the 'matplotlib.pyplot' module and the 'st.pyplot()' function.
if 'Pie Chart' in plot_list:
    st.subheader("Pie Chart")
    pie_data = census_df[['income','gender']].value_counts()
    explode=np.linspace(0,0.25,4)
    plt.figure(figsize = (5, 5))
    plt.title(f"Distribution of records for {features_list} groups")
    plt.pie(pie_data, labels = pie_data.index, autopct = '%1.2f%%', 
           explode=explode,startangle = 30)
    st.pyplot() 
