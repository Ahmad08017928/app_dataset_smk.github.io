import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.title("Syarat Penerimaan pip Kelas 12 ")
st.subheader("Tahun 2023")

st.set_option('deprecation.showPyplotGlobalUse', False)
option = st.sidebar.selectbox('Silahkan Pilih :', ('Home', 'Klassifikasi', 'Tes Klassifikasi'))

if option == 'Home' or option == '':
    df = pd.read_csv('dataset\data - Untitled spreadsheet - Daftar Peserta Didik.csv')
    df_c = df.copy()
    df_c[['No', 'Nama', 'Rombel Saat Ini', 'NIPD', 'JK', 'NISN',
        'Tempat Lahir', 'Tanggal Lahir', 'NIK', 'Agama', 'Alamat', 'RT',
        'RW', 'Dusun', 'Kelurahan', 'Kecamatan', 'Kode Pos',
        'Jenis Tinggal', 'Alat Transportasi', 'HP', 'E-Mail',
        'Penerima KPS', 'No. KPS', 'Nama ayah', 'Tahun Lahir A',
        'Jenjang Pendidikan A', 'Pekerjaan A', 'Penghasilan A', 'NIK A',
        'Nama ibu', 'Tahun Lahir I', 'Jenjang Pendidikan', 'Pekerjaan I',
        'Penghasilan I', 'NIK I', 'Nama wali', 'Tahun Lahir wali',
        'Jenjang Pendidikan wali', 'Pekerjaan wali', 'Penghasilan wali',
        'NIK wali', 'Penerima KIP', 'Nomor KIP', 'Nama di KIP',
        'Nomor KKS', 'Layak PIP (usulan dari sekolah)', 'Alasan Layak PIP',
        'Kebutuhan Khusus', 'Sekolah Asal', 'Anak ke-berapa', 'Lintang',
        'Bujur', 'No KK', 'Berat Badan', 'Tinggi Badan', 'Lingkar Kepala',
        'Jml. Saudara\nKandung', 'Jarak Rumah\nke Sekolah (KM)']] = df_c[['No', 'Nama', 'Rombel Saat Ini', 'NIPD', 'JK', 'NISN',
        'Tempat Lahir', 'Tanggal Lahir', 'NIK', 'Agama', 'Alamat', 'RT',
        'RW', 'Dusun', 'Kelurahan', 'Kecamatan', 'Kode Pos',
        'Jenis Tinggal', 'Alat Transportasi', 'HP', 'E-Mail',
        'Penerima KPS', 'No. KPS', 'Nama ayah', 'Tahun Lahir A',
        'Jenjang Pendidikan A', 'Pekerjaan A', 'Penghasilan A', 'NIK A',
        'Nama ibu', 'Tahun Lahir I', 'Jenjang Pendidikan', 'Pekerjaan I',
        'Penghasilan I', 'NIK I', 'Nama wali', 'Tahun Lahir wali',
        'Jenjang Pendidikan wali', 'Pekerjaan wali', 'Penghasilan wali',
        'NIK wali', 'Penerima KIP', 'Nomor KIP', 'Nama di KIP',
        'Nomor KKS', 'Layak PIP (usulan dari sekolah)', 'Alasan Layak PIP',
        'Kebutuhan Khusus', 'Sekolah Asal', 'Anak ke-berapa', 'Lintang',
        'Bujur', 'No KK', 'Berat Badan', 'Tinggi Badan', 'Lingkar Kepala',
        'Jml. Saudara\nKandung', 'Jarak Rumah\nke Sekolah (KM)']].fillna(method='pad')
    df_c.isnull().sum()
    data = df_c.dropna()
    st.dataframe(data.head())
    st.write("Dari data tersebut terdapat 409 jenis data kuatitatif dan 59 kolom")
    st.header("Dengan menggunakan data di atas, maka saya akan memvisualisasikan data tersebut")
    
    st.write("Kita buat Pie Chart dan Bar Chart Berdasarkan penghasilan dari seorang ayah")
    # Visulisasi data
    unik = data['Penghasilan A'].unique()
    counts = data['Penghasilan A'].value_counts().reindex(['Rp. 500,000 - Rp. 999,999', 'Kurang dari Rp. 500,000',
    'Tidak Berpenghasilan', 'Rp. 1,000,000 - Rp. 1,999,999',
    'Rp. 2,000,000 - Rp. 4,999,999']).fillna(0).tolist()
    labels = unik
    fig, ax = plt.subplots()
    ax.pie(counts, labels=labels, autopct='%1.1f%%')
    ax.axis('equal')
    plt.title('penghasilan ayah')
    st.pyplot(fig)

    plt.figure(figsize=(10, 7))
    plt.bar(labels, counts, color=['Red', 'green', 'blue', 'orange'])
    plt.title('penghasil ayah')
    plt.ylabel('jumlah')
    plt.xlabel('penghasilan')
    plt.xticks(size=7)
    plt.yticks(size=12)
    st.pyplot()
    st.write("""Berdasarkan diagram pie kita dapat disimpulkan bahwa mayoritas orang tua murid di SMK Negeri 1 Kraksaan memiliki 
             penghasilan dalam kisaran 500.000 hingga 999.999, dengan porsi yang mendominasi. Penghasilan 
             1.000.000 hingga 1.999.999 juga cukup signifikan sebagai kontributor kedua terbesar. Sebagian 
             kecil tidak memiliki penghasilan, sedangkan penghasilan kurang dari 500.000 dan 2.000.000 hingga 
             4.999.999 memiliki kontribusi yang lebih kecil.""")
    st.write("""mayoritas orang tua murid di SMK Negeri 1 Kraksaan memiliki penghasilan dalam kisaran 500.000 
             hingga 999.999, dengan tingkat partisipasi yang paling tinggi. Penghasilan 1.000.000 hingga 1.999.999 
             juga memiliki kontribusi yang signifikan sebagai kedua terbesar. Sedangkan kelompok penghasilan tidak 
             berpenghasilan, kurang dari 500.000, dan 2.000.000 hingga 4.999.999 memiliki tingkat partisipasi yang 
             lebih rendah secara berturut-turut.""")
    st.write("Kita buat Pie Chart dan Bar Chart Berdasarkan penghasilan dari seorang ibu")
    
    data_I = data['Penghasilan I'].unique()
    count = data['Penghasilan I'].value_counts().reindex(
    ['Rp. 500,000 - Rp. 999,999','Rp. 1,000,000 - Rp. 1,999,999',
    'Kurang dari Rp. 500,000','Tidak Berpenghasilan',
    'Rp. 2,000,000 - Rp. 4,999,999']).fillna(0).tolist()
    
    labels = data_I
    fig, ax = plt.subplots()
    ax.pie(count, labels=labels, autopct='%1.1f%%')
    ax.axis('equal')
    plt.title('penghasilan ibu')
    st.pyplot(fig)

    plt.figure(figsize=(10, 7))
    plt.bar(labels, count, color=['Red', 'green', 'blue', 'orange'])
    plt.title('penghasilan ibu')
    plt.ylabel('jumlah')
    plt.xlabel('kategori')
    plt.xticks(size=7)
    plt.yticks(size=12)
    st.pyplot()
    st.write("""Berdasarkan diagram bar dan pie, terlihat bahwa mayoritas ibu murid di SMK Negeri 1 Kraksaan 
             memiliki penghasilan kurang dari 500.000, dengan tingkat partisipasi yang tinggi. Penghasilan 1.000.000 
             hingga 1.999.999 juga cukup signifikan sebagai kontributor kedua terbesar, namun, dari diagram pie, terlihat 
             bahwa distribusi ini lebih merata. Sementara itu, kelompok tidak berpenghasilan dan penghasilan 500.000 hingga 999.999 
             memiliki kontribusi yang lebih kecil. Penghasilan 2.000.000 hingga 4.999.999 memiliki tingkat partisipasi yang rendah, 
             menunjukkan bahwa sebagian kecil ibu murid memiliki penghasilan di kisaran tersebut. Dalam perbandingan dengan penghasilan 
             lainnya, kelompok ini memiliki dampak yang lebih terbatas.""")
    
    st.write("Kita buat Pie Chart dan Bar Chart Berdasarkan jenis tempat tinggal")
    data_J = data['Jenis Tinggal'].unique()
    hitung = data['Jenis Tinggal'].value_counts().reindex(
    ['Pesantren','Bersama orang tua', 'Lainnya', 'Wali', 'Asrama']).fillna(0).tolist()
        
    labels = data_J
    fig, ax = plt.subplots()
    ax.pie(hitung, labels=labels, autopct='%1.1f%%')
    ax.axis('equal')
    plt.title('jenis tinggal')
    st.pyplot(fig)

    plt.figure(figsize=(10, 7))
    plt.bar(labels, hitung, color=['Red', 'green', 'blue', 'orange'])
    plt.title('jenis tinggal')
    plt.ylabel('jumlah')
    plt.xlabel('kategori')
    plt.xticks(size=7)
    plt.yticks(size=12)
    st.pyplot()
    st.write("""Berdasarkan diagram bar dan pie, dapat disimpulkan bahwa mayoritas 
             murid di SMK Negeri 1 Kraksaan tinggal bersama orang tua, dengan tingkat 
             partisipasi yang mendominasi. Jenis tempat tinggal bersama wali juga cukup 
             signifikan sebagai kontributor kedua terbesar. Di sisi lain, tingkat partisipasi 
             jenis tempat tinggal di pesantren, tempat tinggal lainnya, dan di asrama relatif 
             lebih rendah, menunjukkan bahwa mayoritas murid lebih memilih untuk tinggal bersama 
             orang tua atau wali mereka.""")
    
    st.write("Visualisasi data dengan Bar Chart tentang Jumlah pengguna alat transportasi berdasarkan kelas")
    plt.figure(figsize=(10, 7))
    plt.bar(data['Alat Transportasi'], data['Rombel Saat Ini'], color=['red', 'green', 'blue', 'orange'])
    plt.title('Jumlah pengguna alat transportasi berdasarkan kelas')
    plt.ylabel('kelas')
    plt.xlabel('Alat transportasi')
    plt.xticks(rotation=45, ha='right', size=10)  
    plt.yticks(size=12)
    st.pyplot()
    st.write("""Jumlah pengguna alat transportasi sepeda, angkutan umum, kendaraan pribadi, jalan kaki, dan sepeda 
             motor relatif sama di semua kelas, menunjukkan bahwa jenis alat transportasi ini digunakan secara 
             merata oleh siswa di berbagai kelas.""")
    st.write("""Alat transportasi ojek memiliki jumlah yang lebih rendah dibandingkan dengan jenis alat transportasi 
             lainnya, menunjukkan bahwa penggunaan ojek sebagai sarana transportasi kurang populer di kalangan siswa.""")
    st.write("""Kesimpulan ini memberikan gambaran umum tentang preferensi alat transportasi siswa berdasarkan kelas, 
             yang dapat digunakan untuk merancang kebijakan transportasi sekolah atau memahami kebutuhan infrastruktur 
             transportasi di sekitar area sekolah.""")
    
    st.write("Visualisasi Data dengan Bar Chart Berdasarkan jenis pekerjaan ayah murid smkn 1 kraksaan")
    data_A = data['Pekerjaan A'].unique()
    hitung_ayah = data['Pekerjaan A'].value_counts().reindex(
    ['Wiraswasta', 'Buruh', 'Nelayan', 'Petani', 'Karyawan Swasta',
       'Tidak bekerja', 'Lainnya', 'Pedagang Kecil', 'Sudah Meninggal',
       'Pedagang Besar', 'PNS/TNI/Polri', 'Peternak', 'Wirausaha']).fillna(0).tolist()
    
    labels = data_A
    plt.figure(figsize=(10, 7))
    plt.bar(labels, hitung_ayah, color=['Red', 'green', 'blue', 'orange'])
    plt.title('Jenis jenis pekerjaan ayah seorang murid di smk negeri 1 kraksaan')
    plt.ylabel('jumlah')
    plt.xlabel('jenis pekerjaan')
    plt.xticks(size=7, rotation = 'vertical')
    plt.yticks(size=12)
    st.pyplot()
    st.write("""Pekerjaan petani menempati posisi ketiga dalam hal jumlah, yang menunjukkan bahwa 
             pertanian masih menjadi mata pencaharian penting di daerah tersebut.""")
        

    
elif option == 'Klassifikasi':
    st.title("Meaching Learning")
    st.write("""Kita tes apakah dari data tersebut seseorang masuk dalam kategori Memerima PIP atau tidak ? Berdasarkan
                Jenis Tinggal, Penerima KPS, Pekerjaan Ayah, Penghasilan Ayah, Pekerjaan Ibu, Penghasilan Ibu, 
                alat Transportasi, Penerima KIP. Dengan metode Classification neural network, Random Forest
                dan metode K-Nearest Neighbor""")
    df = pd.read_csv('dataset\data - Untitled spreadsheet - Daftar Peserta Didik.csv')
    df_c = df.copy()
    df_c[['No', 'Nama', 'Rombel Saat Ini', 'NIPD', 'JK', 'NISN',
        'Tempat Lahir', 'Tanggal Lahir', 'NIK', 'Agama', 'Alamat', 'RT',
        'RW', 'Dusun', 'Kelurahan', 'Kecamatan', 'Kode Pos',
        'Jenis Tinggal', 'Alat Transportasi', 'HP', 'E-Mail',
        'Penerima KPS', 'No. KPS', 'Nama ayah', 'Tahun Lahir A',
        'Jenjang Pendidikan A', 'Pekerjaan A', 'Penghasilan A', 'NIK A',
        'Nama ibu', 'Tahun Lahir I', 'Jenjang Pendidikan', 'Pekerjaan I',
        'Penghasilan I', 'NIK I', 'Nama wali', 'Tahun Lahir wali',
        'Jenjang Pendidikan wali', 'Pekerjaan wali', 'Penghasilan wali',
        'NIK wali', 'Penerima KIP', 'Nomor KIP', 'Nama di KIP',
        'Nomor KKS', 'Layak PIP (usulan dari sekolah)', 'Alasan Layak PIP',
        'Kebutuhan Khusus', 'Sekolah Asal', 'Anak ke-berapa', 'Lintang',
        'Bujur', 'No KK', 'Berat Badan', 'Tinggi Badan', 'Lingkar Kepala',
        'Jml. Saudara\nKandung', 'Jarak Rumah\nke Sekolah (KM)']] = df_c[['No', 'Nama', 'Rombel Saat Ini', 'NIPD', 'JK', 'NISN',
        'Tempat Lahir', 'Tanggal Lahir', 'NIK', 'Agama', 'Alamat', 'RT',
        'RW', 'Dusun', 'Kelurahan', 'Kecamatan', 'Kode Pos',
        'Jenis Tinggal', 'Alat Transportasi', 'HP', 'E-Mail',
        'Penerima KPS', 'No. KPS', 'Nama ayah', 'Tahun Lahir A',
        'Jenjang Pendidikan A', 'Pekerjaan A', 'Penghasilan A', 'NIK A',
        'Nama ibu', 'Tahun Lahir I', 'Jenjang Pendidikan', 'Pekerjaan I',
        'Penghasilan I', 'NIK I', 'Nama wali', 'Tahun Lahir wali',
        'Jenjang Pendidikan wali', 'Pekerjaan wali', 'Penghasilan wali',
        'NIK wali', 'Penerima KIP', 'Nomor KIP', 'Nama di KIP',
        'Nomor KKS', 'Layak PIP (usulan dari sekolah)', 'Alasan Layak PIP',
        'Kebutuhan Khusus', 'Sekolah Asal', 'Anak ke-berapa', 'Lintang',
        'Bujur', 'No KK', 'Berat Badan', 'Tinggi Badan', 'Lingkar Kepala',
        'Jml. Saudara\nKandung', 'Jarak Rumah\nke Sekolah (KM)']].fillna(method='pad')
    data = df_c.dropna()
    # Trasformasi Data
    Numerik = LabelEncoder()
    x = data[['Jenis Tinggal', 'Alat Transportasi', 'Penerima KPS', 'Pekerjaan A', 'Penghasilan A', 'Pekerjaan I', 'Penghasilan I', 
            'Penerima KIP']]
    y = data['Layak PIP (usulan dari sekolah)']
    x['Jenis Tinggal_n'] = Numerik.fit_transform(x['Jenis Tinggal'])
    x['Penerima KPS_n'] = Numerik.fit_transform(x['Penerima KPS'])
    x['Pekerjaan A_n'] = Numerik.fit_transform(x['Pekerjaan A'])
    x['Pekerjaan I_n'] = Numerik.fit_transform(x['Pekerjaan I'])
    x['Alat Transportasi_n'] = Numerik.fit_transform(x['Alat Transportasi'])
    x['Penerima KIP_n'] = Numerik.fit_transform(x['Penerima KIP'])
    x['Penghasilan A_n'] = Numerik.fit_transform(x['Pekerjaan A'])
    x['Penghasilan I_n'] = Numerik.fit_transform(x['Penghasilan I'])
    x_d = x.drop(['Jenis Tinggal', 'Alat Transportasi', 'Penerima KPS', 'Pekerjaan A',
              'Penghasilan A', 'Pekerjaan I', 'Penghasilan I', 'Penerima KIP'], axis='columns')
    
    # training 75% dan testing 25%
    x_train, x_test, y_train, y_test = train_test_split(x_d, y, test_size=0.25, random_state=5000)
    
    # Classification Random Forest
    st.header("1. Random Forest")
    Class_RandomForest = RandomForestClassifier()
    Class_RandomForest.fit(x_train, y_train)
    y_pred_rf = Class_RandomForest.predict(x_test)
    akurasi_RandomFores = accuracy_score(y_test, y_pred_rf)
    st.write(classification_report(y_test, y_pred_rf))
    st.write('Dari pengujian menggunakan metode Random Forest mendapatkan akurasi sebesar **%d persen**'%(akurasi_RandomFores*100))

    # Classification Neural network
    st.header("2. Neural Network")
    from sklearn.neural_network import MLPClassifier
    Class_NN = MLPClassifier()
    Class_NN.fit(x_train, y_train)
    y_pred_NN = Class_NN.predict(x_test)
    akurasi_NN = accuracy_score(y_test, y_pred_NN)
    st.write(classification_report(y_test, y_pred_NN))
    st.write('Dari pengujian menggunakan metode Neural Network mendapatkan akurasi sebesar **%d persen**'%(akurasi_NN*100))
    
    st.header("3. K-Nearest Neighbor")
    # Classification knn
    classification = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classification.fit(x_train, y_train)
    y_pred_knn = classification.predict(x_test)
    a_k = classification_report(y_test, y_pred_knn)
    akurasi = accuracy_score(y_test, y_pred_knn)
    st.write(a_k)
    st.write('Dari pengujian menggunakan metode K-Nearest Neighbor mendapatkan akurasi sebesar **%d persen**'%(akurasi*100))
    # # Classification SVM
    # st.header("3. Support Vector Machine")
    # clf = SVC(kernel='linear')
    # clf.fit(x_train, y_train)
    # y_pred_svm = clf.predict(x_test)
    # akurasi_svm = accuracy_score(y_test, y_pred_svm)
    # st.write(classification_report(y_test, y_pred_svm))
    # st.write('Dari pengujian menggunakan metode Support Vector Machine mendapatkan akurasi sebesar **%d persen**'%(akurasi_svm*100))

elif option == 'Tes Klassifikasi':
    st.title("menklasifikasi apakah seorang murid masuk kategori penerima pip atau tidak ?")
    df = pd.read_csv('dataset\data - Untitled spreadsheet - Daftar Peserta Didik.csv')
    df_c = df.copy()
    df_c[['No', 'Nama', 'Rombel Saat Ini', 'NIPD', 'JK', 'NISN',
        'Tempat Lahir', 'Tanggal Lahir', 'NIK', 'Agama', 'Alamat', 'RT',
        'RW', 'Dusun', 'Kelurahan', 'Kecamatan', 'Kode Pos',
        'Jenis Tinggal', 'Alat Transportasi', 'HP', 'E-Mail',
        'Penerima KPS', 'No. KPS', 'Nama ayah', 'Tahun Lahir A',
        'Jenjang Pendidikan A', 'Pekerjaan A', 'Penghasilan A', 'NIK A',
        'Nama ibu', 'Tahun Lahir I', 'Jenjang Pendidikan', 'Pekerjaan I',
        'Penghasilan I', 'NIK I', 'Nama wali', 'Tahun Lahir wali',
        'Jenjang Pendidikan wali', 'Pekerjaan wali', 'Penghasilan wali',
        'NIK wali', 'Penerima KIP', 'Nomor KIP', 'Nama di KIP',
        'Nomor KKS', 'Layak PIP (usulan dari sekolah)', 'Alasan Layak PIP',
        'Kebutuhan Khusus', 'Sekolah Asal', 'Anak ke-berapa', 'Lintang',
        'Bujur', 'No KK', 'Berat Badan', 'Tinggi Badan', 'Lingkar Kepala',
        'Jml. Saudara\nKandung', 'Jarak Rumah\nke Sekolah (KM)']] = df_c[['No', 'Nama', 'Rombel Saat Ini', 'NIPD', 'JK', 'NISN',
        'Tempat Lahir', 'Tanggal Lahir', 'NIK', 'Agama', 'Alamat', 'RT',
        'RW', 'Dusun', 'Kelurahan', 'Kecamatan', 'Kode Pos',
        'Jenis Tinggal', 'Alat Transportasi', 'HP', 'E-Mail',
        'Penerima KPS', 'No. KPS', 'Nama ayah', 'Tahun Lahir A',
        'Jenjang Pendidikan A', 'Pekerjaan A', 'Penghasilan A', 'NIK A',
        'Nama ibu', 'Tahun Lahir I', 'Jenjang Pendidikan', 'Pekerjaan I',
        'Penghasilan I', 'NIK I', 'Nama wali', 'Tahun Lahir wali',
        'Jenjang Pendidikan wali', 'Pekerjaan wali', 'Penghasilan wali',
        'NIK wali', 'Penerima KIP', 'Nomor KIP', 'Nama di KIP',
        'Nomor KKS', 'Layak PIP (usulan dari sekolah)', 'Alasan Layak PIP',
        'Kebutuhan Khusus', 'Sekolah Asal', 'Anak ke-berapa', 'Lintang',
        'Bujur', 'No KK', 'Berat Badan', 'Tinggi Badan', 'Lingkar Kepala',
        'Jml. Saudara\nKandung', 'Jarak Rumah\nke Sekolah (KM)']].fillna(method='pad')
    data = df_c.dropna()
    # Trasformasi Data
    Numerik = LabelEncoder()
    x = data[['Jenis Tinggal', 'Alat Transportasi', 'Penerima KPS', 'Pekerjaan A', 'Penghasilan A', 'Pekerjaan I', 'Penghasilan I', 
            'Penerima KIP']]
    y = data[['Layak PIP (usulan dari sekolah)']]
    x['Jenis Tinggal_n'] = Numerik.fit_transform(x['Jenis Tinggal'])
    x['Penerima KPS_n'] = Numerik.fit_transform(x['Penerima KPS'])
    x['Pekerjaan A_n'] = Numerik.fit_transform(x['Pekerjaan A'])
    x['Pekerjaan I_n'] = Numerik.fit_transform(x['Pekerjaan I'])
    x['Alat Transportasi_n'] = Numerik.fit_transform(x['Alat Transportasi'])
    x['Penerima KIP_n'] = Numerik.fit_transform(x['Penerima KIP'])
    x['Penghasilan A_n'] = Numerik.fit_transform(x['Pekerjaan A'])
    x['Penghasilan I_n'] = Numerik.fit_transform(x['Penghasilan I'])
    y['Layak PIP (usulan dari sekolah)_n'] = Numerik.fit_transform(y['Layak PIP (usulan dari sekolah)'])
    y['Layak PIP (usulan dari sekolah)_n'] = Numerik.fit_transform(y['Layak PIP (usulan dari sekolah)'])
    x_d = x.drop(['Jenis Tinggal', 'Alat Transportasi', 'Penerima KPS', 'Pekerjaan A',
                'Penghasilan A', 'Pekerjaan I', 'Penghasilan I', 'Penerima KIP'], axis='columns')
    y_d = y.drop("Layak PIP (usulan dari sekolah)", axis='columns')
    
    # training 75% dan testing 25%
    x_train, x_test, y_train, y_test = train_test_split(x_d, y_d, test_size=0.25, random_state=5000)
    
    # Classification Random Forest
    Class_RandomForest = RandomForestClassifier()
    Class_RandomForest.fit(x_train, y_train)

    # Classification Neural network
    from sklearn.neural_network import MLPClassifier
    Class_NN = MLPClassifier()
    Class_NN.fit(x_train, y_train)
    
    # Classification knn
    classification = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classification.fit(x_train, y_train)
    # # Classification SVM
    Class_SVM = SVC(kernel='linear', probability=True)
    Class_SVM.fit(x_train, y_train)

    Nama = st.text_input("Nama")
    JT = st.radio("Jenis tempat tinggal", options=('Pesantren', 'Bersama orang tua', 'Lainnya', 'Wali', 'Asrama'))
    Transportasi = st.radio("Alat Transportasi", options=('Sepeda', 'Angkutan umum/bus/pete-pete', 'Kendaraan pribadi',
       'Jalan kaki', 'Lainnya', 'Ojek', 'Sepeda motor'))
    kps = st.radio("Penerima KPS ", options=("YA", "TIDAK"))
    pekerjaan_ayah = st.radio("Pekerjaan Ayah/Bapak ", options=('Wiraswasta', 'Buruh', 'Nelayan', 'Petani', 'Karyawan Swasta',
       'Tidak bekerja', 'Lainnya', 'Pedagang Kecil', 'Sudah Meninggal',
       'Pedagang Besar', 'PNS/TNI/Polri', 'Peternak', 'Wirausaha'))
    penghasilan_ayah = st.radio("Berapa Penghasilan Ayah/Bapak ", options=('Rp. 500,000 - Rp. 999,999', 'Rp. 1,000,000 - Rp. 1,999,999',
       'Kurang dari Rp. 500,000', 'Tidak Berpenghasilan',
       'Rp. 2,000,000 - Rp. 4,999,999'))
    pekerjaan_ibu = st.radio("Pekerjaan Ibu ",options=('Tidak bekerja', 'Buruh', 'Wiraswasta',
       'Sudah Meninggal', 'Karyawan Swasta', 'Petani', 'Pedagang Kecil',
       'Peternak', 'Wirausaha', 'PNS/TNI/Polri', 'Lainnya'))
    penghasilan_ibu  = st.radio("Berapa Penghasilan Ibu ", options=('Rp. 500,000 - Rp. 999,999', 'Tidak Berpenghasilan',
       'Rp. 1,000,000 - Rp. 1,999,999', 'Kurang dari Rp. 500,000',
       'Rp. 2,000,000 - Rp. 4,999,999'))
    kip = st.radio("Penerima KIP ", options=("YA", "TIDAK"))
    
    if st.button('Klassifikasi') :
        mybarrr = st.progress(0)
        if JT == 'Pesantren':
            JT = 3
        elif JT == 'Bersama orang tua':
            JT = 1
        elif JT == 'Lainnya':
            JT = 2
        elif JT == 'Wali':
            JT = 4
        elif JT == 'Asrama':
            JT = 0
        
        Transportasi = 0
        if Transportasi =='Sepeda':
            Transportasi = 5
        elif Transportasi == 'Angkutan umum/bus/pete-pete':
            Transportasi = 0
        elif Transportasi == 'Kendaraan pribadi':
            Transportasi = 2
        elif Transportasi == 'Jalan kaki':
            Transportasi = 1
        elif Transportasi == 'Lainnya':
            Transportasi = 3
        elif Transportasi == 'Ojek':
            Transportasi = 4
        elif Transportasi == 'Sepeda motor':
            Transportasi = 6
    
        kps = 0 if kps == 'TIDAK' else 1
        
        pekerjaan_ayah = 0
        if pekerjaan_ayah == 'Wiraswasta':
            pekerjaan_ayah = 11
        elif pekerjaan_ayah == 'Buruh':
            pekerjaan_ayah = 0
        elif pekerjaan_ayah == 'Nelayan':
            pekerjaan_ayah = 3
        elif pekerjaan_ayah == 'Petani':
            pekerjaan_ayah = 7
        elif pekerjaan_ayah == 'Karyawan Swasta':
            pekerjaan_ayah = 1
        elif pekerjaan_ayah == 'Tidak bekerja':
            pekerjaan_ayah = 10
        elif pekerjaan_ayah == 'Lainnya':
            pekerjaan_ayah = 2
        elif pekerjaan_ayah == 'Pedagang Kecil':
            pekerjaan_ayah = 6
        elif pekerjaan_ayah == 'Sudah Meninggal':
            pekerjaan_ayah = 9
        elif pekerjaan_ayah == 'Pedagang Besar':
            pekerjaan_ayah = 5
        elif pekerjaan_ayah == 'PNS/TNI/Polri':
            pekerjaan_ayah = 4
        elif pekerjaan_ayah == 'Peternak':
            pekerjaan_ayah = 8
        elif pekerjaan_ayah == 'Wirausaha':
            pekerjaan_ayah = 12
    
        penghasilan_ayah = 0
        if penghasilan_ayah == 'Rp. 500,000 - Rp. 999,999':
            penghasilan_ayah = 11
        elif penghasilan_ayah == 'Rp. 1,000,000 - Rp. 1,999,999':
            penghasilan_ayah = 0
        elif penghasilan_ayah == 'Kurang dari Rp. 500,000':
            penghasilan_ayah = 3
        elif penghasilan_ayah == 'Tidak Berpenghasilan':
            penghasilan_ayah = 7
        elif penghasilan_ayah == 'Rp. 2,000,000 - Rp. 4,999,999':
            penghasilan_ayah = 1
        
        pekerjaan_ibu = 0
        if pekerjaan_ibu == 'Lainnya':
            pekerjaan_ibu = 2
        elif pekerjaan_ibu == 'Tidak bekerja':
            pekerjaan_ibu = 8
        elif pekerjaan_ibu == 'Buruh':
            pekerjaan_ibu = 0 
        elif pekerjaan_ibu == 'Wiraswasta':
            pekerjaan_ibu = 9 
        elif pekerjaan_ibu == 'Sudah Meninggal':
            pekerjaan_ibu = 7
        elif pekerjaan_ibu == 'Karyawan Swasta':
            pekerjaan_ibu = 1
        elif pekerjaan_ibu == 'Petani':
            pekerjaan_ibu = 5
        elif pekerjaan_ibu == 'Pedagang Kecil':
            pekerjaan_ibu = 4
        elif pekerjaan_ibu == 'Peternak':
            pekerjaan_ibu = 6
        elif pekerjaan_ibu == 'Wirausaha':
            pekerjaan_ibu = 10
        elif pekerjaan_ibu == 'PNS/TNI/Polri':
            pekerjaan_ibu = 3
        
        penghasilan_ibu = 0
        if penghasilan_ibu == 'Rp. 500,000 - Rp. 999,999':
            penghasilan_ibu = 3
        elif penghasilan_ibu == 'Tidak Berpenghasilan':
            penghasilan_ibu = 4
        elif penghasilan_ibu == 'Rp. 1,000,000 - Rp. 1,999,999':
            penghasilan_ibu = 1
        elif penghasilan_ibu == 'Kurang dari Rp. 500,000':
            penghasilan_ibu = 0
        elif penghasilan_ibu == 'Rp. 2,000,000 - Rp. 4,999,999':
            penghasilan_ibu = 2
        
        kip = 0 if kip == 'TIDAK' else 1

# Konversi input pengguna menjadi nilai numerik
    # JT_numeric = Numerik.transform([JT])[0] if JT in Numerik.classes_ else -1
    # Transportasi_numeric = Numerik.transform([Transportasi])[0] if Transportasi in Numerik.classes_ else -1
    # pekerjaan_ayah_numeric = Numerik.transform([pekerjaan_ayah])[0] if pekerjaan_ayah in Numerik.classes_ else -1
    # penghasilan_ayah_numeric = Numerik.transform([penghasilan_ayah])[0] if penghasilan_ayah in Numerik.classes_ else -1
    # pekerjaan_ibu_numeric = Numerik.transform([pekerjaan_ibu])[0] if pekerjaan_ibu in Numerik.classes_ else -1
    # penghasilan_ibu_numeric = Numerik.transform([penghasilan_ibu])[0] if penghasilan_ibu in Numerik.classes_ else -1
    # kip_numeric = Numerik.transform([kip])[0] if kip in Numerik.classes_ else -1

# ...

    # tes
    # tes = [[JT_numeric, Transportasi_numeric, kps, pekerjaan_aya
    # h_numeric, penghasilan_ayah_numeric,
    #         pekerjaan_ibu_numeric, penghasilan_ibu_numeric, kip_numeric]]
    
        tes = [[JT, Transportasi, kps, pekerjaan_ayah, penghasilan_ayah, pekerjaan_ibu, penghasilan_ibu, kip]]
        
        hasil_RandomForest = Class_RandomForest.predict(tes)
        akurasi_RandomForest = Class_RandomForest.predict_proba(tes)
        hasil_NN = Class_NN.predict(tes)
        akurasi_NN = Class_NN.predict_proba(tes)
        hasil_knn = classification.predict(tes)
        akurasi_knn = classification.predict_proba(tes)
        hasil_svm = Class_SVM.predict(tes)
        akurasi_svmm = Class_SVM.predict_proba(tes)
        
        for persen in range (100):
            time.sleep(0.01)
            mybarrr.progress(persen+1)
        
        st.subheader("Random Forest")
        if hasil_RandomForest [0] == 1:
            st.write("{} Termasuk dalam kategori penerima pip, dengan akurasi prediksi {}".format(Nama, round(akurasi_RandomForest[0][hasil_RandomForest[0]]*100), 3)) 
        else :
            st.write("{} tidak termasuk dalam kategori penerima pip, dengan akurasi prediksi {}".format(Nama, round(akurasi_RandomForest[0][hasil_RandomForest[0]]*100), 3))
            
        st.subheader("Neural Network")
        if hasil_NN [0] == 1:
            st.write("{} Termasuk dalam kategori penerima pip, dengan akurasi prediksi {}".format(Nama, round(akurasi_NN[0][hasil_NN[0]]*100), 3)) 
        else :
            st.write("{} tidak termasuk dalam kategori penerima pip, dengan akurasi prediksi {}".format(Nama, round(akurasi_NN[0][hasil_NN[0]]*100), 3)) 
        
        st.subheader("K-Nearest Neighbor")
        if hasil_knn [0] == 1:
            st.write("{} Termasuk dalam kategori penerima pip, dengan akurasi prediksi {}".format(Nama, round(akurasi_knn[0][hasil_knn[0]]*100), 3)) 
        else :
            st.write("{} tidak termasuk dalam kategori penerima pip, dengan akurasi prediksi {}".format(Nama, round(akurasi_knn[0][hasil_knn[0]]*100), 3)) 
                
        st.subheader("Support Vector Machine")
        if hasil_svm [0] == 1:
            st.write("{} Termasuk dalam kategori penerima pip, dengan akurasi prediksi {}".format(Nama, round(akurasi_svmm[0][hasil_svm[0]]*100), 3)) 
        else :
            st.write("{} tidak termasuk dalam kategori penerima pip, dengan akurasi prediksi {}".format(Nama, round(akurasi_svmm[0][hasil_svm[0]]*100), 3)) 
                
