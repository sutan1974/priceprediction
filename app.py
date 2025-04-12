import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

# Set page config to optimize for full-screen layout and add logo
st.set_page_config(page_title="Prediksi Harga Properti", layout="wide", page_icon="logo.png")

# Display the logo above the title
st.image('logo.png', width=200)  # Adjust width as needed

# Introduction
st.title("Prediksi Harga & Rekomendasi Tempat Penginapan Kamu!💰🏨")

st.markdown("""
    Selamat datang di aplikasi **Prediksi dan Rekomendasi Harga Properti & Hotel**! 
    Aplikasi ini memungkinkan Anda untuk memprediksi harga properti dan mendapatkan rekomendasi properti 
    berdasarkan fitur-fitur seperti jumlah kamar tidur, lokasi, tipe properti, dan lainnya. 
    Cukup masukkan informasi properti Anda di bawah ini, lalu klik tombol **Prediksi Harga** untuk melihat estimasi harga, 
    serta rekomendasi properti yang sesuai dengan kisaran harga yang Anda inginkan.
""")

# Construct file paths using os.path.join
data_path = os.path.join(os.getcwd(), 'listings.csv')
model_path = os.path.join(os.getcwd(), 'random_search.joblib')

# Load the data
listings_df = pd.read_csv(data_path)

# Load the model
model = joblib.load(model_path)

# Define numeric and categorical columns
categorical_cols = ['room_type', 'host_is_superhost', 'neighbourhood', 'property_type']

# Fill missing categorical values with mode
for col in categorical_cols:
    if col in listings_df.columns:
        mode_val = listings_df[col].mode()
        if not mode_val.empty:
            listings_df[col] = listings_df[col].fillna(mode_val[0])

# Buat tiga kolom input dengan layout responsif
col1, col2, col3 = st.columns([1, 1, 1])  # Tiga kolom dengan lebar yang sama

# Kolom 1: Fitur Properti
with col1:
    st.subheader("🏠 Fitur Properti")
    bedrooms = st.number_input('Jumlah Kamar Tidur', min_value=0, max_value=10, value=1)
    bathrooms = st.number_input('Jumlah Kamar Mandi', min_value=0, max_value=10, value=1)
    beds = st.number_input('Jumlah Tempat Tidur', min_value=0, max_value=10, value=1)
    minimum_nights = st.number_input('Minimum Menginap (malam)', min_value=1, max_value=30, value=1)
    maximum_nights = st.number_input('Maksimum Menginap (malam)', min_value=1, max_value=365, value=30)

# Kolom 2: Informasi Lokasi
with col2:
    st.subheader("📍 Informasi Lokasi")
    availability_365 = st.number_input('Ketersediaan (hari dalam setahun)', min_value=0, max_value=365, value=365)
    review_scores_rating = st.number_input('Skor Ulasan (0-100)', min_value=0, max_value=100, value=80)
    reviews_per_month = st.number_input('Ulasan per Bulan', min_value=0.0, max_value=100.0, value=1.0)
    latitude = st.number_input('Lintang (Latitude)', min_value=-90.0, max_value=90.0, value=0.0)
    longitude = st.number_input('Bujur (Longitude)', min_value=-180.0, max_value=180.0, value=0.0)

# Kolom 3: Informasi Tambahan
with col3:
    st.subheader("📋 Informasi Tambahan")
    room_type = st.selectbox('Tipe Kamar', sorted(listings_df['room_type'].unique()))
    host_is_superhost = st.selectbox('Host adalah Superhost?', ['Yes', 'No'])
    neighbourhood = st.selectbox('Lingkungan', sorted(listings_df['neighbourhood'].unique()))
    property_type = st.selectbox('Tipe Properti', sorted(listings_df['property_type'].unique()))


# Fungsi untuk memproses input
def process_input():
    # Map categorical and binary variables
    host_is_superhost_binary = 1 if host_is_superhost == 'Yes' else 0

    # Buat DataFrame input tunggal
    input_data = pd.DataFrame([{
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'beds': beds,
        'minimum_nights': minimum_nights,
        'maximum_nights': maximum_nights,
        'availability_365': availability_365,
        'review_scores_rating': review_scores_rating,
        'reviews_per_month': reviews_per_month,
        'room_type': room_type,
        'host_is_superhost': host_is_superhost_binary,
        'neighbourhood': neighbourhood,
        'latitude': latitude,
        'longitude': longitude,
        'property_type': property_type
    }])

    # Apply Label Encoding to categorical columns (same as training process)
    label_encoder = LabelEncoder()
    
    categorical_columns = ['room_type', 'neighbourhood', 'property_type']
    for col in categorical_columns:
        input_data[col] = label_encoder.fit_transform(input_data[col])

    # Fill missing values if any
    input_data = input_data.fillna(0)

    # Align the input data to match the model's feature columns
    if hasattr(model, 'feature_names_in_'):
        model_columns = model.feature_names_in_
    else:
        model_columns = input_data.columns  # Assuming input_data has the correct features

    # Reindex the input data to match the model's columns, and fill missing columns with zeros
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    return input_data

# Before prediction: Check the input data for any missing values
input_data = process_input()

# Check if there are any missing values right before prediction
if input_data.isnull().sum().sum() > 0:
    input_data = input_data.fillna(0)  # Ensure no missing values

# Prediction
if st.button('Check it Now (Price & Hotels)'):
    # Run the prediction
    prediction = model.predict(input_data)

    # Highlight the predicted price
    predicted_price = prediction[0]
    st.markdown(f"<h3 style='color: #2E8B57; font-weight: bold;'>Estimasi Harga Hotel Kamu: ${predicted_price:,.2f}</h3>", unsafe_allow_html=True)

    # Load the listings dataframe (you might want to load it from your dataset or a file)

    # Remove currency symbols and convert 'price' to numeric in listings_df
    listings_df['price'] = listings_df['price'].replace('[\$,]', '', regex=True).astype(float)

    # Group by 'listing_url' and 'name' and get the mean of 'price'
    listings_df = listings_df.groupby(['listing_url', 'picture_url','name'])['price'].mean().reset_index()

    # Get the price range based on the predicted value (±10% for example)
    price_lower_bound = predicted_price * 0.9
    price_upper_bound = predicted_price * 1.1

    # Filter listings within the price range
    recommendations = listings_df[(listings_df['price'] >= price_lower_bound) & (listings_df['price'] <= price_upper_bound)]

    # Sort by price (mean) and get top 10 recommendations
    top_recommendations = recommendations.sort_values(by='price', ascending=True).head(10)

    # Display the top 10 recommendations in two columns
    st.markdown("Rekomendasi Properti Berdasarkan Harga (range harga berdasarkan nilai prediksi (±10%)): ")

    # Create two columns
    col1, col2 = st.columns(2)

    # Loop through the top recommendations and display them in the columns
    for index, row in top_recommendations.iterrows():
        # Determine which column to place the item in
        if index % 2 == 0:  # Even index will go into the first column
            column = col1
        else:  # Odd index will go into the second column
            column = col2

        # Check if the 'picture_url' column exists and has a valid URL
        with column:
            if 'picture_url' in top_recommendations.columns and pd.notna(row['picture_url']):
                # Display the image in the selected column with a fixed width (you can adjust the width value as needed)
                st.image(row['picture_url'], width=200)  # Set the width of the image to a fixed value (200px in this case)

                # Display the listing name and link
                st.markdown(f"**{row['name']}**: [Link to Airbnb]({row['listing_url']})")
            else:
                # If no image is available, display the name and the link
                st.markdown(f"**{row['name']}**: Image not available")
                st.markdown(f"[Link to Airbnb]({row['listing_url']})")


