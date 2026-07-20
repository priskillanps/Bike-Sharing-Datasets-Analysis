import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (assuming the data is already downloaded and unzipped in the local directory)
@st.cache
def load_data():
    df_day = pd.read_csv('day.csv')
    df_hour = pd.read_csv('hour.csv')
    return df_day, df_hour

# Load datasets
df_day, df_hour = load_data()

# Streamlit app
st.title("Proyek Analisis Data: Bike Sharing Dataset")
st.write("**Nama:** Priskilla Novianna Puteri Br Silalahi")
st.write("**Email:** priskillanps@gmail.com")
st.write("**ID Dicoding:** priskillanps")

# Display dataset info
st.subheader("Data Wrangling")
st.write("### Gathering Data")
st.write("Tampilan beberapa data dari dataset harian (df_day):")
st.dataframe(df_day.head())

# Dataset information
st.write("Informasi dataset harian (df_day):")
st.write(df_day.info())

# Display column statistics
st.write("Deskripsi statistik dataset harian (df_day):")
st.write(df_day.describe())

# Check missing values
st.write("Jumlah missing values di dataset harian (df_day):")
st.write(df_day.isna().sum())

# Data Cleaning
st.subheader("Cleaning Data")
st.write("Pembersihan data dilakukan terhadap outlier di variabel target cnt_hourly dan cnt_daily.")

# Outliers visualization for cnt_hourly
st.write("Visualisasi boxplot untuk outliers (cnt_hourly):")
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_hour['cnt'], color='#9fa8f0')
plt.title('Boxplot of Hourly Count')
st.pyplot(plt)

# Outliers visualization for cnt_daily
st.write("Visualisasi boxplot untuk outliers (cnt_daily):")
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_day['cnt'], color='#9fa8f0')
plt.title('Boxplot of Daily Count')
st.pyplot(plt)

# Exploratory Data Analysis
st.subheader("Exploratory Data Analysis (EDA)")

st.write("Distribusi data numerik:")
for column in df_day.columns:
    if pd.api.types.is_numeric_dtype(df_day[column]):
        plt.figure(figsize=(8, 6))
        sns.histplot(df_day[column], color='#9fa8f0', kde=True)
        plt.title(f'Distribution of {column}')
        st.pyplot(plt)

# Correlation matrix
st.write("Matriks Korelasi:")
correlation_matrix = df_day.corr(numeric_only=True)
plt.figure(figsize=(22, 20))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Bike Sharing Dataset')
st.pyplot(plt)

# Insights
st.subheader("Insight:")
st.write("""
- Pengguna terdaftar (registered) lebih banyak menyumbang terhadap total penyewaan sepeda dibandingkan pengguna kasual.
- Temperatur memiliki pengaruh yang positif terhadap jumlah penyewaan, sementara cuaca buruk menurunkan permintaan.
- Musim juga berpengaruh signifikan terhadap permintaan penyewaan sepeda, dengan musim panas atau semi memiliki penyewaan lebih tinggi.
""")

# Visualizations for question 1: Demand by season and weather
st.subheader("Pertanyaan 1: Bagaimana pola permintaan sepeda berdasarkan musim dan cuaca?")
seasonal_demand = df_day.groupby('season')['cnt'].mean()

# Barplot for seasonal demand
plt.figure(figsize=(8, 5))
sns.barplot(x=seasonal_demand.index, y=seasonal_demand.values, palette="Blues")
plt.title('Rata-rata Permintaan Sepeda Berdasarkan Musim')
st.pyplot(plt)

# Visualizations for question 2: Demand by hour
st.subheader("Pertanyaan 2: Apa faktor utama yang mempengaruhi lonjakan penyewaan sepeda pada jam-jam tertentu?")
hourly_demand = df_hour.groupby('hr')['cnt'].mean()

# Lineplot for hourly demand
plt.figure(figsize=(10, 6))
sns.lineplot(x=hourly_demand.index, y=hourly_demand.values, marker="o")
plt.title('Rata-rata Permintaan Sepeda Berdasarkan Jam')
st.pyplot(plt)

# Conclusion
st.subheader("Conclusion")
st.write("""
- **Kesimpulan Pertanyaan 1:** Pola trend permintaan sepeda berada di puncaknya saat cuaca cerah pada musim gugur dan menurun jika cuaca semakin buruk.
- **Kesimpulan Pertanyaan 2:** Faktor utama yang mempengaruhi lonjakan penyewaan sepeda adalah perjalanan ke/dari tempat kerja pada hari kerja dan aktivitas rekreasi pada hari libur. Temperatur juga berperan dalam keputusan untuk bersepeda.
""")
