import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from math import log

# Fungsi untuk memuat data dari file yang diunggah
def load_data(file):
    if file.name.endswith('csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('xls', 'xlsx')):
        return pd.read_excel(file)
    else:
        st.error("Format file tidak didukung.")
        return None

def preprocessing(status_pesanan):
    # Daftar kolom-kolom yang akan dihapus
    if 'Status Pesanan' in status_pesanan.columns:
        status_pesanan.drop(columns=['Status Pesanan'], inplace=True)

    columns_to_drop = ['Source.Name', 'Username (Pembeli)', 'Nama Penerima', 'No. Telepon', 'Alamat Pengiriman',
                       'No. Pesanan', 'Status Pembatalan/ Pengembalian', 'No. Resi', 'Catatan', 'Catatan dari Pembeli',
                       'Waktu Pesanan Selesai', 'Antar ke counter/ pick-up', 'Waktu Pengiriman Diatur',
                       'Returned quantity']

    # Menghapus kolom-kolom tersebut dari DataFrame
    resi = status_pesanan['No. Resi']
    status_pesanan = status_pesanan.drop(columns=columns_to_drop, errors='ignore')

    for i, x in status_pesanan.iterrows():
        if str(status_pesanan['Waktu Pembayaran Dilakukan'][i]) == 'nan':
            status_pesanan['Waktu Pembayaran Dilakukan'][i] = status_pesanan['Waktu Pesanan Dibuat'][i]

    # Penggantian dengan nilai tertentu
    status_pesanan['Alasan Pembatalan'].fillna('Tidak Diketahui', inplace=True)
    status_pesanan['Pesanan Harus Dikirimkan Sebelum (Menghindari keterlambatan)'].fillna('Tidak Diketahui',
                                                                                          inplace=True)
    status_pesanan['Metode Pembayaran'].fillna('Metode Pembayaran Lainnya', inplace=True)
    status_pesanan['SKU Induk'].fillna(0, inplace=True)
    status_pesanan['Nomor Referensi SKU'].fillna(9999, inplace=True)
    status_pesanan['Nama Variasi'].fillna('Variasi Tidak Diketahui', inplace=True)
    status_pesanan['Total Pembayaran'].fillna(0, inplace=True)
    status_pesanan['Total Harga Produk'].fillna(0, inplace=True)

    # Menghapus karakter koma atau titik dan mengonversi ke tipe data float
    status_pesanan['Total Pembayaran'] = status_pesanan['Total Pembayaran'].replace('[\.,]', '', regex=True).astype(
        float)

    # Mengonversi ke tipe data integer
    status_pesanan['Total Pembayaran'] = status_pesanan['Total Pembayaran'].astype(int)

    # Penggantian dengan nilai rata-rata kolom
    status_pesanan['Metode Pembayaran'].fillna(status_pesanan['Metode Pembayaran'].mode()[0], inplace=True)
    # Penggantian dengan nilai median kolom
    status_pesanan['SKU Induk'].fillna(status_pesanan['SKU Induk'].median(), inplace=True)

    # Forward fill untuk kolom waktu
    status_pesanan['Waktu Pembayaran Dilakukan'].fillna(method='ffill', inplace=True)

    # Menghapus baris dengan missing values pada kolom 'Total Pembayaran'
    status_pesanan.dropna(subset=['Total Pembayaran'], inplace=True)

    # Categorical Variable
    categoric_variable = ['Alasan Pembatalan', 'Opsi Pengiriman',
                          'Pesanan Harus Dikirimkan Sebelum (Menghindari keterlambatan)', 'Waktu Pesanan Dibuat',
                          'Waktu Pembayaran Dilakukan', 'Metode Pembayaran', 'Nama Produk', 'Nomor Referensi SKU',
                          'Nama Variasi', 'Berat Produk', 'Total Berat', 'Paket Diskon', 'Kota/Kabupaten', 'Provinsi']

    # Encode the categorical variables
    for col in categoric_variable:
        status_pesanan[col] = status_pesanan[col].astype(str)  # Konversi ke tipe data string
        le = LabelEncoder()
        status_pesanan[col] = le.fit_transform(status_pesanan[col])

    for col in status_pesanan.columns:
        if col in categoric_variable:
            continue
        status_pesanan[col] = status_pesanan[col].apply(lambda x: log(x) if x > 0 else 0)

    return resi, status_pesanan


# Fungsi untuk melakukan prediksi
def predict_status(data, scaler, model):
    # Pra-pemrosesan data (misalnya: normalisasi menggunakan scaler)
    resi, data = preprocessing(data)
    data_scaled = scaler.transform(data)  # Skalakan data

    # Prediksi menggunakan model SVM
    predictions = model.predict(data_scaled)

    return resi, predictions

def show_sidebar():
    st.sidebar.header('Deskripsi Fitur yang Mempengaruhi Pembatalan Transaksi')
    feature_descriptions = [
        ("Total Pembayaran:", "Total yang harus dibayar oleh pembeli."),
        ("Ongkos Kirim yang Dibayar Pembeli:", "Biaya pengiriman yang ditanggung oleh pembeli."),
        ("Metode Pembayaran:", "Metode pembayaran yang dipilih oleh pembeli."),
        ("Voucher Ditanggung Penjual:", "Nilai voucher yang ditanggung oleh penjual."),
        ("Estimasi Potongan Biaya Pengiriman:", "Estimasi biaya pengiriman."),
        ("Opsi Pengiriman:", "Metode pengiriman yang dipilih oleh pembeli."),
        ("Waktu Pembayaran Dilakukan:", "Waktu pembeli melakukan pembayaran.")
    ]
    for feature, description in feature_descriptions:
        st.sidebar.write(f"- **{feature}** {description}")

# Fungsi untuk membagi DataFrame menjadi halaman-halaman
@st.cache_data(show_spinner=False)
def split_dataframe(input_df, rows):
    df_pages = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
    return df_pages


# Bagian utama aplikasi
def main():
    st.title('Prediksi Pembatalan Transaksi')

    # Tampilkan sidebar
    show_sidebar()

    # Upload file CSV atau Excel
    uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            # Load scaler dari file
            scaler = joblib.load('scaler.pkl')
            # Load model dari file
            model = joblib.load('model.pkl')

            # Prediksi status pesanan
            resi, predictions = predict_status(data, scaler, model)

            # Tampilkan hasil prediksi dengan pagination
            st.subheader('Hasil Prediksi')
            # Buat DataFrame untuk menampilkan hasil prediksi
            result_df = pd.DataFrame({'Resi': resi, 'Status': predictions})
            result_df['Status'] = result_df['Status'].replace({0: 'Selesai', 1: 'Batal'})

            total_transactions = len(result_df)
            successful_transactions = len(result_df[result_df["Status"] == "Selesai"])
            canceled_transactions = len(result_df[result_df["Status"] == "Batal"])

            # Menampilkan informasi dalam bentuk card
            col1, col2, col3 = st.columns(3)

            with col1:
                container = st.container(height=170)
                container.subheader("ℹ️ ")
                container.write("**Jumlah Transaksi**")
                container.subheader(total_transactions)

            with col2:
                container = st.container(height=170)
                container.subheader("✅️ ")
                container.write("**Sukses**")
                container.subheader(successful_transactions)

            with col3:
                container = st.container(height=170)
                container.subheader("❌️ ")
                container.write("**Batal**")
                container.subheader(canceled_transactions)

            # Menu bagian atas
            top_menu = st.columns(3)
            with top_menu[0]:
                sort = st.radio("Urutkan Data", options=["Ya", "Tidak"], index=1)
            if sort == "Ya":
                with top_menu[1]:
                    sort_field = st.selectbox("Urutkan berdasarkan", options=result_df.columns)
                with top_menu[2]:
                    sort_direction = st.radio("Arah", options=["⬆️", "⬇️"])
                result_df = result_df.sort_values(by=sort_field, ascending=sort_direction == "⬆️", ignore_index=True)

            # Menu bagian bawah
            bottom_menu = st.columns((4, 1, 1))
            with bottom_menu[2]:
                batch_size = st.selectbox("Ukuran Halaman", options=[10, 25, 50, 100])
            with bottom_menu[1]:
                total_pages = (len(result_df) // batch_size) + 1 if len(result_df) % batch_size != 0 else len(
                    result_df) // batch_size
                current_page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1)
            with bottom_menu[0]:
                st.markdown(f"Halaman **{current_page}** dari **{total_pages}**")

            # Memecah DataFrame menjadi halaman-halaman
            pages = split_dataframe(result_df, batch_size)

            # Menampilkan tabel pada halaman yang dipilih
            st.dataframe(pages[current_page - 1], use_container_width=True)

# Jalankan aplikasi
if __name__ == '__main__':
    main()
