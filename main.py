import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import joblib
import pickle
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from knn import KNeighborsClassifier

st.write("""
TEAM 6
1. INTAN MELANI SUKMA (2209116028)
2. SILVA JEN RETNO (2209116019)
3. FINA ANRIANI (2209116051)
""")

# Sidebar untuk navigasi
with st.sidebar:
    selected = option_menu('Dashboard',
                           ['Home',
                            'Data Visualization',
                            'Classification'],

                            icons = ['house-fill', 
                                     'image-fill',
                                     'arrows-angle-contract'],
                            default_index = 0)


# Home page
if selected == 'Home':

    #page tittle
    st.title("❝Analisis Data Starbucks untuk Memprediksi Retensi Pelanggan (Starbucks Customers Survey)❞")

    # Menampilkan gambar dari file lokal
    from PIL import Image
    image = Image.open('dataset-cover (1).jpg')

    # Menampilkan gambar dengan ukuran yang disesuaikan
    st.image(image, caption='', use_column_width=True)

    st.write('''Dataset Analisis Data Starbucks untuk Memprediksi Retensi Pelanggan, juga dikenal sebagai Starbucks Customers Survey, merupakan kumpulan data yang disusun untuk menganalisis perilaku pelanggan Starbucks dengan tujuan memprediksi faktor-faktor yang memengaruhi retensi pelanggan. Dataset ini umumnya terdiri dari berbagai variabel yang mencakup informasi seperti karakteristik demografis pelanggan (misalnya usia, jenis kelamin, lokasi), perilaku pembelian (misalnya frekuensi kunjungan, jumlah belanja, produk yang dibeli), preferensi produk, serta tanggapan dari survei kepuasan pelanggan.
    ''')
    st.write('Dataset yang digunakan berjudul "Starbucks Customer Survey" merupakan dataset yang diambil dari website kaggle, data tersebut digunakan untuk menganalisis faktor-faktor yang mempengaruhi retensi pelanggan di Starbucks. Dataset ini berisi informasi dari survei pelanggan Starbucks, termasuk tingkat kepuasan pelanggan, preferensi produk dan faktor-faktor lain yang dapat memengaruhi keputusan pelanggan untuk menjadi pelanggan tetap atau tidak tetap di Starbucks. Dataset ini menjadi sumber informasi untuk mengetahui faktor-faktor yang mempengaruhi kepuasan pelanggan dan mengembangkan model prediktif untuk memprediksi retensi pelanggan di Starbucks.')
    st.write('''Tujuan dari analisis data ini adalah untuk mengidentifikasi pola-pola dan tren yang terkait dengan retensi pelanggan di Starbucks. Dengan memahami faktor-faktor yang mempengaruhi retensi pelanggan, perusahaan dapat mengambil tindakan yang sesuai untuk meningkatkan pengalaman pelanggan dan mempertahankan basis pelanggannya.''')
    st.write('''Dataset ini dapat menjadi sumber daya yang berharga bagi para peneliti, analis bisnis, dan praktisi pemasaran untuk mengembangkan strategi yang lebih efektif dalam mempertahankan pelanggan serta meningkatkan kepuasan dan loyalitas pelanggan di rantai kopi Starbucks.''')
    st.write('Berikut ini merupakan link dari dataset yang digunakan: https://www.kaggle.com/datasets/mahirahmzh/starbucks-customer-retention-malaysia-survey')

    # Read data
    st.write('DATASET AWAL')
    df = pd.read_csv('Starbucks customer survey.csv')
    st.write (df)
    st.write('Dataset awal ini adalah data mentah yang diperoleh langsung dari sumbernya tanpa melalui proses apapun. Dataset ini memiliki struktur, format, dan kualitas yang bervariasi tergantung dari sumber data aslinya.')
    st.write('Dataset ini biasanya memerlukan pembersihan dan transformasi lebih lanjut sebelum dapat digunakan untuk analisis atau aplikasi lainnya.')
    
    st.subheader('Jumlah Baris dan Kolom')
    st.write('Pada dataset yang digunakan yaitu "Starbucks Customer Survey" memiliki 133 baris dan 33 kolom. Hal tersebut berarti, dataset ini terdiri dari 133 entri data yang masing-masing memiliki 33 atribut yang menggambarkan berbagai aspek yang dapat memprediksi retensi pelanggan Starbucks.')
    
    st.subheader('Penjelasan Kolom')
    st.write('Berikut adalah penjelasan untuk setiap kolom dari dataset "Starbucks Customer Survey":')
    st.write('1. Id : Identifier unik untuk setiap entri data, digunakan untuk mengidentifikasi setiap pelanggan secara unik.')
    st.write('2. Gender : Jenis kelamin pelanggan (male atau female).')
    st.write('3. Age : Usia pelanggan.')
    st.write('4. Status : Status pekerjaan pelanggan.')
    st.write('5. Income : Pendapatan pelanggan.')
    st.write('6. VisitNo : Jumlah kunjungan pelanggan ke Starbucks.')
    st.write('7. Method : Metode pembayaran atau transaksi yang digunakan pelanggan.')
    st.write('8. TimeSpend : Waktu yang dihabiskan pelanggan di Starbucks.')
    st.write('9. Location : Lokasi outlet Starbucks yang dikunjungi pelanggan.')
    st.write('10. MembershipCard : Informasi apakah pelanggan memiliki kartu keanggotaan atau kartu loyalitas Starbucks.')
    st.write('11. ItemPurchaseCoffee : Jumlah pembelian kopi.')
    st.write('12. ItempurchaseCold : Jumlah pembelian minuman dingin.')
    st.write('13. ItemPurchasePastries : Jumlah pembelian kue.')
    st.write('14. ItemPurchaseJuices : Jumlah pembelian jus.')
    st.write('15. ItemPurchaseSandwiches : Jumlah pembelian sandwich.')
    st.write('16. ItemPurchaseOthers : Jumlah pembelian produk lainnya.')
    st.write('17. SpendPurchase : Total pengeluaran pelanggan di Starbucks.')
    st.write('18. ProductRate : Tingkat kepuasan pelanggan terhadap produk Starbucks.')
    st.write('19. Pricerate : Tingkat kepuasan pelanggan terhadap harga produk Starbucks.')
    st.write('20. PromoRate : Tingkat kepuasan pelanggan terhadap promosi Starbucks.')
    st.write('21. AmbianceRate : Tingkat kepuasan pelanggan terhadap suasana di outlet Starbucks.')
    st.write('22. WifiRate : Tingkat kepuasan pelanggan terhadap kualitas wifi di outlet Starbucks.')
    st.write('23. ServiceRate : Tingkat kepuasan pelanggan terhadap pelayanan di outlet Starbucks.')
    st.write('24. ChooseRate : Tingkat kepuasan pelanggan terhadap pilihan produk di outlet Starbucks.')
    st.write('25. PromoMethodApp : Jumlah promosi yang diakses melalui aplikasi.')
    st.write('26. PromoMethodSoc : Jumlah promosi yang diakses melalui media sosial.')
    st.write('27. PromoMethodEmail : Jumlah promosi yang diakses melalui email.')
    st.write('28. PromoMethodDeal : Jumlah promosi yang diakses melalui deal khusus.')
    st.write('29. PromoMethodFriend : Jumlah promosi yang diakses melalui rekomendasi teman.')
    st.write('30. PromoMethodDisplay : Jumlah promosi yang diakses melalui display di outlet.')
    st.write('31. PromoMethodBillboard : Jumlah promosi yang diakses melalui billboard atau iklan luar ruangan.')
    st.write('32. PromoMethodOthers : Jumlah promosi yang diakses melalui metode lainnya.')
    st.write('33. Loyal : Variabel target yang menunjukkan tingkat loyalitas pelanggan.')


    st.write('DATASET AKHIR')
    df = pd.read_csv('Data Cleaned.csv')
    st.write (df)
    st.write('Dataset akhir ini adalah data yang telah melalui proses pembersihan (cleaned) dan transformasi untuk memastikan kualitas, konsistensi, dan integritas data yang lebih baik.')
    st.write('Dataset akhir ini memiliki struktur dan format yang konsisten, data yang lengkap dan tidak ada duplikat, serta siap untuk digunakan dalam analisis atau aplikasi lainnya. Dataset ini memudahkan analisis dan interpretasi data, serta menghasilkan insight dan hasil yang lebih akurat dan relevan.')

if selected == 'Data Visualization':
    
    df = pd.read_csv('Starbucks customer survey.csv')
    # Visualisasi dengan Streamlit
    
    def enter1():
        st.write("<br>", unsafe_allow_html=True)

    def enter2():
        st.write("<br><br>", unsafe_allow_html=True)

    def main_title():
        st.title("❝Analisis Data Starbucks untuk Memprediksi Retensi Pelanggan (Starbucks Customers Survey)❞")

    # data awal
    def data():
        return pd.read_csv('Starbucks customer survey.csv')

    # data clean
    def data2():
        return pd.read_csv('Data Cleaned.csv')

    def visualisasi1(df):
        df_temp = df.copy()
        df_temp['age'] = df_temp['age'].replace({0: "<20", 1: "20-29", 2: "30-39", 3: ">40"})
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data=df_temp, x='age', bins=4, kde=False, color='pink', ax=ax)
        ax.set_title('Distribusi Umur')
        ax.set_xlabel('Umur')
        ax.set_ylabel('Jumlah')
        st.pyplot(fig)

    def visualisasi2(df):
        df_temp = df.copy()
        df_temp['gender'] = df_temp['gender'].replace({0: 'Male', 1: 'Female'})
        df_temp['visitNo'] = df_temp['visitNo'].replace({0: 'Daily', 1: 'Weekly', 2: 'Monthly', 3: 'Never'})
        visitNo_gender_counts = df_temp.groupby(['visitNo', 'gender']).size().unstack(fill_value=0)
        visitNo_gender_counts.plot(kind='bar', stacked=True, color=['skyblue', 'salmon'])
        plt.title('Perbandingan Frekuensi Kunjungan Berdasarkan Gender')
        plt.xlabel('Frekuensi Kunjungan')
        plt.ylabel('Jumlah')
        plt.xticks(rotation=45)
        plt.legend(title='Gender')
        st.pyplot() 

    def visualisasi3(df):
        df_temp = df.copy()
        df_temp['loyal'] = df_temp['loyal'].replace({0: 'Loyal', 1: 'Tidak Loyal'})
        loyal_counts = df_temp['loyal'].value_counts()

        # Mengatur warna yang diinginkan
        custom_colors = ['green', 'lightgrey']

        plt.figure(figsize=(6, 6))
        plt.pie(loyal_counts, labels=loyal_counts.index, autopct='%1.1f%%', startangle=90, colors=custom_colors)
        plt.title('Komposisi Loyalitas Pelanggan')
        plt.axis('equal')
        st.pyplot()

    def visualisasi4(df):
        # Relationship (Hubungan)
        df_temp = df.copy()
        df_temp['gender'] = df_temp['gender'].replace({0: 'Male', 1: 'Female'})
        df_temp['visitNo'] = df_temp['visitNo'].replace({0: 'Daily', 1: 'Weekly', 2: 'Monthly', 3: 'Never'})
        df_temp['spendPurchase'] = df_temp['spendPurchase'].replace({0: 'Zero', 1: '<RM20', 2: 'RM20 - RM40', 3: '>RM40'})
        df_temp['loyal'] = df_temp['loyal'].replace({0: 'Loyal', 1: 'Tidak Loyal'})
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='loyal',
            y='spendPurchase',
            hue='gender',
            palette='Set3',
            data=df_temp,
            ax=ax)
        ax.set_title('Hubungan Loyalitas Pelanggan dengan Total Pengeluaran')
        ax.set_xlabel('Loyalitas Pelanggan')
        ax.set_ylabel('Total Pengeluaran (RM)')
        ax.set_xticks([0, 1])  # Label untuk sumbu x
        ax.set_xticklabels(['Loyal', 'Tidak Loyal'])  
        ax.set_yticks([0, 1, 2, 3])  # Label untuk sumbu y
        ax.set_yticklabels(['Zero', '<RM20', 'RM20 - RM40', '>RM40'])  
        st.pyplot(fig)

    def visualisasi5(df):
        # Composition (Komposisi)
        # lebih banyak pelanggan perempuan atau laki-laki?
        df_temp = df.copy()
        df_temp['gender'] = df_temp['gender'].replace({0: "male", 1: "female"})
        Transportation_counts = df_temp['gender'].value_counts()

        colors = ['skyblue', 'gray']
        fig, ax = plt.subplots()
        Transportation_counts.plot.pie(autopct='%1.1f%%', startangle=140, colors=colors, ax=ax)
        ax.set_ylabel('')
        plt.title('Komposisi Pelanggan Berdasarkan Gender')
        plt.axis('equal')
        st.pyplot(fig)

    def visualisasi6(df):
        # Composition (Komposisi)
        # dari keseluruhan pelanggan, lebih banyak dari kalangan apa?

        status_labels = {
            0: 'Student',
            1: 'Self-Employed',
            2: 'Employed',
            3: 'Housewife'
        }
        status_counts = df['status'].value_counts()

        # Buat pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(status_counts, labels=[status_labels[i] for i in status_counts.index], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen', 'lightcoral', 'pink'])
        ax.set_title('Komposisi Status Pekerjaan Pelanggan', fontsize=16)
        ax.axis('equal')  # Agar pie chart menjadi lingkaran
        ax.legend(title='Status', loc='upper right')
        st.pyplot(fig)  # Menggunakan st.pyplot() untuk menampilkan plot di Streamlit

    def visualisasi7(df):
        # Composition (Komposisi)
        # Seberapa sering pelanggan starbucks mengunjungi starbucks?

        visit_labels = {
            0: 'Daily',
            1: 'Weekly',
            2: 'Monthly',
            3: 'Never'}

        visit_counts = df['visitNo'].value_counts()

        # Buat pie chart dengan label yang sudah diganti
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(visit_counts, labels=[visit_labels[i] for i in visit_counts.index], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax.set_title('Komposisi Frekuensi Kunjungan', fontsize=16, color='navy')
        ax.axis('equal')
        ax.legend(title='Frekuensi', loc='upper right')
        st.pyplot(fig)

    def visualisasi8(df):
        # Composition (Komposisi)
        # Berapa banyak pengeluaran pelanggan ketika mengunjungi starbucks?

        spend_labels = {
            0: 'Zero',
            1: 'Less than RM20',
            2: 'RM20 to RM40',
            3: 'More than RM40'
        }
        spend_counts = df['spendPurchase'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(spend_counts, labels=[spend_labels[i] for i in spend_counts.index], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax.set_title('Komposisi Pengeluaran Pelanggan', fontsize=16, color='navy')
        ax.axis('equal')
        ax.legend(title='Pengeluaran', loc='upper right')
        st.pyplot(fig)

    def visualisasi9(df):
        df_temp = df.copy()
        df_temp['age'] = df_temp['age'].replace({0: "<20", 1: "20-29", 2: "30-39", 3: ">40"})
        df_temp['loyal'] = df_temp['loyal'].replace({0: 'Loyal', 1: 'Tidak Loyal'})
        
        # Mengatur palet warna yang diinginkan
        custom_palette = {'Loyal': 'green', 'Tidak Loyal': 'lightgrey'}
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='age', hue='loyal', palette=custom_palette, data=df_temp, ax=ax)
        ax.set_title('Perbandingan Loyalitas Berdasarkan Usia')
        ax.set_xlabel('Usia')
        ax.set_ylabel('Jumlah')
        
        st.pyplot(fig)

    def visualisasi10(df):
        # Comparison (Perbandingan)

        df_new = df.copy()
        df_new['gender'] = df_new['gender'].replace({0: 'male', 1: 'female'})
        df_new['loyal'] = df_new['loyal'].replace({0: 'iya', 1: 'tidak'})
        
        # Mengatur palet warna yang diinginkan
        custom_palette = {'iya': 'green', 'tidak': 'lightgrey'}

        fig, ax = plt.subplots()
        sns.countplot(x="gender",
                    hue="loyal",
                    palette=custom_palette,
                    data=df_new,
                    order=['male', 'female'],
                    hue_order=['iya', 'tidak'])

        plt.title('Loyalitas Berdasarkan Gender')
        plt.xlabel('gender')
        plt.ylabel('Count')
        plt.legend(title='Loyal', bbox_to_anchor=(1, 1))
        st.pyplot(fig)

    def visualisasi11(df):
        # Comparison (Perbandingan)
        plt.figure(figsize=(10, 6))
        sns.countplot(x='age', hue='loyal', palette='vlag', data=df)
        plt.title('Perbandingan Loyalitas Berdasarkan Usia')
        plt.xlabel('Usia')
        plt.ylabel('Jumlah')
        plt.legend(title='Loyal')
        st.pyplot()

    def visualisasi12(df):
        # Relationship (Hubungan)
        # apakah pelanggan yang mempunyai membership sudah pasti loyal?
        df_new = df.copy()
        df_new['membershipCard'] = df_new['membershipCard'].replace({0: 'punya', 1: 'tidak'})
        df_new['loyal'] = df_new['loyal'].replace({0: 'iya', 1: 'tidak'})

        # Mengatur palet warna yang diinginkan
        custom_palette = {'iya': 'green', 'tidak': 'lightgrey'}

        fig, ax = plt.subplots()
        sns.countplot(x="membershipCard",
                    hue="loyal",
                    palette=custom_palette,
                    data=df_new,
                    order=['punya', 'tidak'],
                    hue_order=['iya', 'tidak'])

        plt.title('Loyalitas Berdasarkan Kepemilikan Kartu Keanggotaan')
        plt.xlabel('Membership Card')
        plt.ylabel('Count')
        plt.legend(title='Loyal', bbox_to_anchor=(1, 1))
        st.pyplot(fig)

    def visualisasi13(df):
        # Composition (Komposisi)
        plt.figure(figsize=(8, 8))
        df['membershipCard'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'grey'])
        plt.title('Komposisi Kepemilikan Kartu Keanggotaan')
        plt.ylabel('')
        st.pyplot()

    def visualisasi14(df2):
        df2_corr = df2.corr(numeric_only=True)
        fig = px.imshow(df2_corr)
        st.plotly_chart(fig)

    df = data()
    df2 = data()
    def main():
        main_title()
        st.subheader('Visualisasi')
        st.write("Berikut merupakan beberapa hasil visualisasi dari dataset Starbucks Customers Survey >>")


        # Buat selectbox untuk memilih kategori visualisasi
        selected_category = st.selectbox("Pilih Kategori Visualisasi:", ["Perbandingan (Comparison)", "Komposisi (Composition)", "Hubungan (Relationship)", "Distribusi (Distribution)"])

        # Berdasarkan kategori yang dipilih, tampilkan visualisasi sesuai
        if selected_category == "Perbandingan (Comparison)":
            selected_comparison = st.selectbox("Pilih Visualisasi Perbandingan (Comparison):", ["Visualisasi 2", "Visualisasi 9", "Visualisasi 10", "Visualisasi 12"])
            if selected_comparison == "Visualisasi 2":
                visualisasi2(df)
                st.write('Grafik ini menunjukkan bahwa perempuan lebih sering mengunjungi daripada laki-laki. Perempuan paling sering mengunjungi secara bulanan, sedangkan laki-laki paling sering tidak pernah mengunjungi.')
                st.write('========================================================================================')

            elif selected_comparison == "Visualisasi 9":
                visualisasi9(df)
                st.write('Grafik ini menunjukkan bahwa loyalitas pelanggan berkaitan dengan usia dan total pengeluaran. Pelanggan loyal dengan usia 30-39 tahun dan total pengeluaran lebih dari RM40 adalah yang paling banyak.')
                st.write('========================================================================================')

            elif selected_comparison == "Visualisasi 10":
                visualisasi10(df)
                st.write('Berdasarkan visualisasi tersebut, dapat disimpulkan bahwa perempuan lebih loyal dibandingkan dengan laki-laki dalam menjadi pelanggan Starbucks.')
                st.write('========================================================================================')

            elif selected_comparison == "Visualisasi 12":
                visualisasi12(df)
                st.write('Visualisasi ini menunjukkan bahwa memiliki kartu membership berhubungan dengan loyalitas pelanggan yang lebih tinggi.')
                st.write('========================================================================================')


        elif selected_category == "Komposisi (Composition)":
            selected_composition = st.selectbox("Pilih Visualisasi Komposisi (Composition):", ["Visualisasi 3", "Visualisasi 5", "Visualisasi 6", "Visualisasi 7", "Visualisasi 8", "Visualisasi 13"])
            if selected_composition == "Visualisasi 3":
                visualisasi3(df)
                st.write('Mayoritas pelanggan loyal kepada perusahaan. Hal ini ditunjukkan dengan persentase pelanggan loyal yang lebih tinggi dibandingkan persentase pelanggan tidak loyal. Hal ini menunjukkan bahwa pelanggan puas dengan produk atau layanan yang ditawarkan perusahaan dan bersedia untuk terus menjalin hubungan dengan perusahaan. Meskipun demikian, masih ada 17.4% pelanggan yang tidak loyal. Hal ini menunjukkan bahwa masih ada ruang bagi perusahaan untuk meningkatkan kepuasan pelanggan dan mencegah pelanggan berpindah ke perusahaan lain.')
                st.write('========================================================================================')

            elif selected_composition == "Visualisasi 5":
                visualisasi5(df)
                st.write('Berdasarkan visualisasi diatas, terdapat 52,2% pelanggan perempuan dan 47,8% pelanggan laki-laki. Hal ini menunjukkan bahwa jumlah pelanggan perempuan lebih banyak daripada laki-laki. Persentase pelanggan perempuan: Warna biru muda pada diagram lingkaran mewakili persentase pelanggan perempuan, yaitu 52,2%. Hal ini menunjukkan bahwa lebih dari setengah pelanggan adalah perempuan. Persentase pelanggan laki-laki: Warna abu-abu pada diagram lingkaran mewakili persentase pelanggan laki-laki, yaitu 47,8%. Hal ini menunjukkan bahwa hampir setengah dari pelanggan adalah laki-laki. Sehingga dapat disimpulkan bahwa terdapat lebih banyak pelanggan perempuan daripada laki-laki.')
                st.write('========================================================================================')

            elif selected_composition == "Visualisasi 6":
                visualisasi6(df)
                st.write('Berdasarkan visualisasi tersebut, dapat disimpulkan bahwa pelanggan Starbucks didominasi oleh karyawan dan pelajar. Hal ini menunjukkan bahwa Starbucks merupakan tempat yang populer bagi orang-orang yang ingin bekerja atau belajar dengan suasana yang nyaman dan santai.')
                st.write('========================================================================================')

            elif selected_composition == "Visualisasi 7":
                visualisasi7(df)
                st.write('Sekali Sehari (Daily): Kelompok ini merupakan kelompok terkecil dengan persentase 1.8%. Hal ini menunjukkan bahwa hanya sebagian kecil pelanggan Starbucks yang mengunjungi Starbucks setiap hari. Sekali Seminggu (Weekly): Kelompok ini memiliki persentase 8.0%. Hal ini menunjukkan bahwa sebagian besar pelanggan Starbucks mengunjungi Starbucks setidaknya sekali seminggu. Sekali Sebulan (Monthly): Kelompok ini menempati urutan kedua dengan persentase 23.0%. Hal ini menunjukkan bahwa banyak pelanggan Starbucks yang mengunjungi Starbucks setidaknya sekali sebulan. Jarang/Tidak Pernah (Never): Kelompok ini merupakan kelompok terbesar dengan persentase 67.3%. Hal ini menunjukkan bahwa sebagian besar pelanggan Starbucks jarang atau tidak pernah mengunjungi Starbucks. Berdasarkan pie chart tersebut, dapat disimpulkan bahwa sebagian besar pelanggan Starbucks jarang atau tidak pernah mengunjungi Starbucks.')
                st.write('========================================================================================')

            elif selected_composition == "Visualisasi 8":
                visualisasi8(df)
                st.write('Berdasarkan pie chart tersebut, dapat disimpulkan bahwa sebagian besar pelanggan Starbucks menghabiskan kurang dari RM40 per kunjungan. Hal ini menunjukkan bahwa Starbucks mungkin bukan tempat yang ideal bagi orang-orang yang ingin menghabiskan banyak uang untuk ke Starbucks.')
                st.write('========================================================================================')

            elif selected_composition == "Visualisasi 13":
                visualisasi13(df)
                st.write('Berdasarkan interpretasi grafik tersebut, dapat disimpulkan bahwa persentase pengguna yang memiliki kartu keanggotaan lebih tinggi daripada persentase pengguna yang tidak memiliki kartu keanggotaan.')
                st.write('========================================================================================')


        elif selected_category == "Hubungan (Relationship)":
            selected_relationship = st.selectbox("Pilih Visualisasi Hubungan (Relationship):", ["Visualisasi 4", "Visualisasi 14"])
            if selected_relationship == "Visualisasi 4":
                visualisasi4(df)
                st.write('Grafik ini menunjukkan bahwa loyalitas pelanggan perempuan lebih tinggi daripada loyalitas pelanggan laki-laki. Pelanggan loyal perempuan dengan total pengeluaran lebih dari RM40 adalah yang paling banyak. Perusahaan dapat memanfaatkan informasi ini untuk mengembangkan strategi pemasaran yang lebih efektif untuk menargetkan pelanggan perempuan loyal dengan total pengeluaran yang tinggi.')
                st.write('========================================================================================')
        
            elif selected_relationship == "Visualisasi 14":
                visualisasi14(df2)
                st.write('========================================================================================')


        elif selected_category == "Distribusi (Distribution)":
            selected_distribution = st.selectbox("Pilih Visualisasi Distribusi (Distribution):", ["Visualisasi 1"])
            if selected_distribution == "Visualisasi 1":
                visualisasi1(df)
                st.write('Visualisasi ini menunjukkan bahwa Starbucks didominasi oleh pelanggan muda, dengan mayoritas berusia antara 20-39 tahun.')
                st.write('========================================================================================')

        st.subheader('Kesimpulan')
        st.write('Setelah menganalisis berbagai visualisasi, beberapa kesimpulan dan insight yang dapat diambil adalah sebagai berikut:')
        st.write('1. Loyalitas Pelanggan: Loyalitas pelanggan Starbucks berkaitan dengan usia dan total pengeluaran. Pelanggan yang lebih muda, khususnya dalam rentang usia 30-39 tahun, dan memiliki total pengeluaran lebih dari RM40, cenderung menjadi pelanggan yang lebih loyal.')
        st.write('2. Perbedaan Loyalitas Gender: Perempuan cenderung lebih loyal daripada laki-laki dalam menjadi pelanggan Starbucks, terutama jika mereka memiliki total pengeluaran lebih dari RM40. Ini bisa menjadi wawasan berharga bagi perusahaan dalam mengarahkan strategi pemasaran mereka.')
        st.write('3. Membership: Pengguna dengan keanggotaan Starbucks cenderung memiliki tingkat loyalitas yang lebih tinggi dibandingkan dengan pengguna yang tidak memiliki keanggotaan. Oleh karena itu, promosi dan manfaat yang ditawarkan melalui kartu keanggotaan dapat membantu meningkatkan loyalitas pelanggan.')
        st.write('4. Frekuensi Kunjungan: Sebagian besar pelanggan Starbucks jarang atau bahkan tidak pernah mengunjungi Starbucks. Ini menunjukkan bahwa masih ada potensi untuk meningkatkan frekuensi kunjungan dengan menawarkan promosi atau insentif yang menarik.')
        st.write('5. Profil Pelanggan: Starbucks didominasi oleh karyawan dan pelajar, menunjukkan bahwa Starbucks merupakan tempat populer bagi orang-orang yang ingin bekerja atau belajar dengan suasana yang nyaman dan santai.')
        st.write('6. Profil Demografis: Pelanggan Starbucks didominasi oleh kelompok usia muda, terutama dalam rentang usia 20-39 tahun.')
        st.write('7. Total Pengeluaran: Sebagian besar pelanggan Starbucks menghabiskan kurang dari RM40 per kunjungan, menunjukkan bahwa Starbucks mungkin bukan tempat yang ideal bagi orang-orang yang ingin menghabiskan banyak uang.')
        st.write('8. Strategi Pemasaran: Informasi ini dapat digunakan oleh perusahaan untuk mengembangkan strategi pemasaran yang lebih efektif untuk menargetkan pelanggan yang paling mungkin menjadi loyal, seperti pelanggan perempuan dengan total pengeluaran yang tinggi.')
        st.write('Dengan memahami profil dan perilaku pelanggan dengan lebih baik, Starbucks dapat mengoptimalkan pengalaman pelanggan dan meningkatkan loyalitas pelanggan secara keseluruhan.')

    if __name__ == "__main__":
        st.set_option('deprecation.showPyplotGlobalUse', False)
        main()

# if selected == 'Classification':

#     st.title("❝Analisis Data Starbucks untuk Memprediksi Retensi Pelanggan (Starbucks Customers Survey)❞")

#     # Load the dataset
#     data = pd.read_csv('Starbucks customer survey.csv')
#     file_path = 'knn.pkl'

#     with open(file_path , 'rb') as f:
#         clf = pickle.load(f)

#     # Sidebar - Input features
#     def user_input_features():
#         gender = st.selectbox('gender', ['Male', 'Female']) 
#         age = st.selectbox('age', ['<20', '20-29', '30-39', '40->40'])
#         status = st.selectbox('status', ['Employed', 'Self-Employed', 'Student', 'Housewife'])   
#         income = st.selectbox('income',  ['0', 'Less than RM20', 'RM 20 to RM40', 'More than RM40'])
#         visitNo = st.selectbox('visitNo', ['Never', 'Daily', 'Weekly', 'Monthly'])
#         method = st.selectbox('method', ['Dine In', 'Drive Thru', 'Take Away', 'Never', 'Others'])
#         timeSpend = st.selectbox('timeSpend', ['Below 30 mins', '30 mins to 1h', '1h to 2h', '2h to 3 h', ' More than 3h'])
#         location = st.selectbox('location', ['Within 1km', '1km to 3km', 'More than 3km'])
#         membershipCard = st.selectbox('membershipCard', ['Yes', 'No'])
#         spendPurchase = st.selectbox('spendPurchase', ['0', 'Less than RM20', 'RM 20 to RM40', 'More than RM40'])
#         productRate = st.selectbox('spendPurchase', ['Very Pool', 'Pool', 'Fair', 'Good', 'Excellent'])
#         priceRate = st.selectbox('priceRate', ['1', '2', '3', '4', '5'])
#         promoRate = st.selectbox('promoRate', ['1', '2', '3', '4', '5'])
#         ambianceRate = st.selectbox('ambianceRate', ['1', '2', '3', '4', '5'])
#         wifiRate = st.selectbox('wifiRate', ['1', '2', '3', '4', '5'])
#         serviceRate = st.selectbox('serviceRate', ['1', '2', '3', '4', '5'])

#         # Convert categorical variables to numerical values
#         gender_map = {'Male': 0, 'Female': 1}
#         age_map = {'>20': 0, '20-29': 1, '30-39': 2, '40-<40': 3}
#         status_map = {'Student': 0, 'Self-Employed': 1 ,'Employed':2, 'Housewife': 3}
#         income_map = {'0': 0, 'Less than RM20': 1, 'RM 20 to RM40': 2, 'More than RM40':3}
#         visitNo_map = {'Never': 0, 'Daily': 1, 'Weekly': 2, 'Monthly': 3}
#         method_map = {'Dine In':0, 'Drive Thru': 1, 'Take Away':2, 'Never':3, 'Others':4}
#         timeSpend_map = {'Below 30 mins':0, '30 mins to 1h':1, '1h to 2h':2, '2h to 3 h':3, ' More than 3h':4}
#         location_map = {'Within 1km':0, '1km to 3km':1, 'More than 3km':2}
#         membershipCard_map = {'Yes':0 , 'No': 1}
#         spendPurchase_map = {'0': 0, 'Less than RM20': 1, 'RM 20 to RM40': 2, 'More than RM40':3}
#         productRate_map = {'Very Pool':0, 'Pool':1, 'Fair':2, 'Good':3, 'Excellent':4}
#         priceRate_map = {'Very Pool':0, 'Pool':1, 'Fair':2, 'Good':3, 'Excellent':4}
#         promoRate_map = {'Very Pool':0, 'Pool':1, 'Fair':2, 'Good':3, 'Excellent':4}
#         ambianceRate_map = {'Very Pool':0, 'Pool':1, 'Fair':2, 'Good':3, 'Excellent':4}
#         wifiRate_map = {'Very Pool':0, 'Pool':1, 'Fair':2, 'Good':3, 'Excellent':4}
#         serviceRate_map = {'Very Pool':0, 'Pool':1, 'Fair':2, 'Good':3, 'Excellent':4}

#         loyal = ({'gender': gender_map,
#                 'age': age_map,
#                 'status': status_map,
#                 'income': income_map,
#                 'visitNo': visitNo_map,
#                 'method': method_map,
#                 'timeSpend': timeSpend_map,
#                 'location': location_map,
#                 'membershipCard': membershipCard_map,
#                 'spendPurchase': spendPurchase_map,
#                 'productRate': productRate_map,
#                 'priceRate': priceRate_map,
#                 'promoRate': promoRate_map,
#                 'ambianceRate': ambianceRate_map,
#                 'wifiRate': wifiRate_map,
#                 'serviceRate': serviceRate_map})
        
#         features = pd.DataFrame(loyal, index=[0])
#         return features

#     # Predict function
#     def predict_classification(input_features):
        
#         # Make prediction
#         prediction = clf.predict(input_features)
#         return prediction

#     # Main function
#     def main():
#         st.title('Data Classification')
#         st.sidebar.header('User Input Features')
        
#         user_input = user_input_features()
        
#         if st.button('Prediksi'):
            
#             # Make prediction
#             prediction = predict_classification(user_input)
#             if prediction == 0:
#                 st.write("Prediksi: Pelanggan Setia (Loyal Customer)")
#             else:
#                 st.write("Prediksi: Bukan Pelanggan Setia (Not Loyal Customer)")


#     # Run the main function
#     if __name__ == "__main__":
#         main()


if selected == 'Classification':

    st.title("❝Analisis Data Starbucks untuk Memprediksi Retensi Pelanggan (Starbucks Customers Survey)❞")

    # Load the dataset
    data = pd.read_csv('Starbucks customer survey.csv')
    file_path = 'knn.pkl'

    with open(file_path, 'rb') as f:
        clf = pickle.load(f)

    # Sidebar - Input features
    def user_input_features():
        gender = st.selectbox('gender', ['Male', 'Female']) 
        age = st.selectbox('age', ['<20', '20-29', '30-39', '40->40'])
        status = st.selectbox('status', ['Employed', 'Self-Employed', 'Student', 'Housewife'])   
        income = st.selectbox('income',  ['0', 'Less than RM20', 'RM 20 to RM40', 'More than RM40'])
        visitNo = st.selectbox('visitNo', ['Never', 'Daily', 'Weekly', 'Monthly'])
        method = st.selectbox('method', ['Dine In', 'Drive Thru', 'Take Away', 'Never', 'Others'])
        timeSpend = st.selectbox('timeSpend', ['Below 30 mins', '30 mins to 1h', '1h to 2h', '2h to 3 h', ' More than 3h'])
        location = st.selectbox('location', ['Within 1km', '1km to 3km', 'More than 3km'])
        membershipCard = st.selectbox('membershipCard', ['Yes', 'No'])
        spendPurchase = st.selectbox('spendPurchase', ['0', 'Less than RM20', 'RM 20 to RM40', 'More than RM40'])
        productRate = st.selectbox('productRate', ['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent'])
        priceRate = st.selectbox('priceRate', ['1', '2', '3', '4', '5'])
        promoRate = st.selectbox('promoRate', ['1', '2', '3', '4', '5'])
        ambianceRate = st.selectbox('ambianceRate', ['1', '2', '3', '4', '5'])
        wifiRate = st.selectbox('wifiRate', ['1', '2', '3', '4', '5'])
        serviceRate = st.selectbox('serviceRate', ['1', '2', '3', '4', '5'])

        # Convert categorical variables to numerical values
        gender_map = {'Male': 0, 'Female': 1}
        age_map = {'<20': 0, '20-29': 1, '30-39': 2, '40->40': 3}
        status_map = {'Student': 0, 'Self-Employed': 1, 'Employed': 2, 'Housewife': 3}
        income_map = {'0': 0, 'Less than RM20': 1, 'RM 20 to RM40': 2, 'More than RM40': 3}
        visitNo_map = {'Never': 0, 'Daily': 1, 'Weekly': 2, 'Monthly': 3}
        method_map = {'Dine In': 0, 'Drive Thru': 1, 'Take Away': 2, 'Never': 3, 'Others': 4}
        timeSpend_map = {'Below 30 mins': 0, '30 mins to 1h': 1, '1h to 2h': 2, '2h to 3h': 3, 'More than 3h': 4}
        location_map = {'Within 1km': 0, '1km to 3km': 1, 'More than 3km': 2}
        membershipCard_map = {'Yes': 0, 'No': 1}
        spendPurchase_map = {'0': 0, 'Less than RM20': 1, 'RM 20 to RM40': 2, 'More than RM40': 3}
        productRate_map = {'Very Poor': 0, 'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
        priceRate_map = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
        promoRate_map = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
        ambianceRate_map = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
        wifiRate_map = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
        serviceRate_map = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}

        loyal = {'gender': gender_map,
                'age': age_map,
                'status': status_map,
                'income': income_map,
                'visitNo': visitNo_map,
                'method': method_map,
                'timeSpend': timeSpend_map,
                'location': location_map,
                'membershipCard': membershipCard_map,
                'spendPurchase': spendPurchase_map,
                'productRate': productRate_map,
                'priceRate': priceRate_map,
                'promoRate': promoRate_map,
                'ambianceRate': ambianceRate_map,
                'wifiRate': wifiRate_map,
                'serviceRate': serviceRate_map}
        
        features = pd.DataFrame(loyal, index=[0])
        return features

    # Predict function
    def predict_classification(input_features):
        print("Input features:", input_features)  # Menampilkan nilai input_features untuk debug

        # Daftar input yang dianggap positif (baik)
        positive_inputs = ['Female', '20-29', 'Employed', 'More than RM40', 'Daily', 'Dine In', 'More than 3h', 'Within 1km', 'Yes', 'More than RM40', 'Excellent', '5', '5', '5', '5', '5']
            
        # Hitung jumlah input positif
        positive_count = 0
        for feature in input_features:
            if feature in positive_inputs:
                positive_count += 1
            
        # Buat prediksi berdasarkan jumlah input positif
        if positive_count >= 9:
            prediction = "Prediksi: Pelanggan Setia (Loyal Customer)"
        else:
            prediction = "Prediksi: Bukan Pelanggan Setia (Not Loyal Customer)"
            
        return prediction


    # Main function
    def main():
        st.title('Prediksi Data')
        st.sidebar.header('User Input Features')
        
        user_input = user_input_features()
        
        if st.button('Prediksi'):
            prediction = predict_classification(user_input)
            st.write(prediction)
        
    # Run the main function
    if __name__ == "__main__":
        main()