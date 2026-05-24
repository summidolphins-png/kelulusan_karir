import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EduMining | Prediksi Kelulusan & Karier Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
<style>
    /* Main Background & Text Colors */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #f3f4f6 !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Custom Card Design */
    .glass-card {
        background-color: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 800;
        color: #6366f1 !important;
    }
    
    /* Recommendations Box */
    .rec-box {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    .rec-success {
        background-color: rgba(16, 185, 129, 0.08);
        border-left: 4px solid #10b981;
        color: #a7f3d0;
    }
    
    .rec-danger {
        background-color: rgba(239, 68, 68, 0.08);
        border-left: 4px solid #ef4444;
        color: #fecaca;
    }
    
    .rec-warning {
        background-color: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #f59e0b;
        color: #fef3c7;
    }
    
    .rec-info {
        background-color: rgba(59, 130, 246, 0.08);
        border-left: 4px solid #3b82f6;
        color: #dbeafe;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS & METRICS ---
@st.cache_resource
def load_models():
    model_g = None
    model_c = None
    if os.path.exists("model_graduation.joblib"):
        model_g = joblib.load("model_graduation.joblib")
    if os.path.exists("model_career.joblib"):
        model_c = joblib.load("model_career.joblib")
    return model_g, model_c

@st.cache_data
def load_metrics():
    if os.path.exists("metrics.json"):
        with open("metrics.json", "r") as f:
            return json.load(f)
    return None

model_graduation, model_career = load_models()
metrics_data = load_metrics()

# --- SIDEBAR BRANDING ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🎓 Edu<span style='color:#6366f1;'>Mining</span></h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.85rem; color:#9ca3af;'>Sistem Klasifikasi Data Mining Modern</p>", unsafe_allow_html=True)
    st.write("---")
    st.info("Aplikasi ini menggunakan algoritma **Random Forest Classifier** yang dilatih menggunakan data historis 5.000 mahasiswa.")
    st.write("---")
    st.markdown("<small style='color:#6b7280; display:block; text-align:center;'>Versi 1.0.0 &copy; 2026</small>", unsafe_allow_html=True)

# --- MAIN TITLE ---
st.title("Sistem Prediksi Kelulusan & Arah Karier Mahasiswa")
st.write("Analisis prediktif kelulusan mahasiswa tepat waktu dan kecocokan arah karier berbasis integrasi data akademik, sosial, dan eksternal.")

# --- NAVIGATION TABS ---
tab_dashboard, tab_single, tab_batch = st.tabs([
    "📊 Dashboard & Statistik", 
    "👤 Simulasi Profil Mahasiswa", 
    "📂 Analisis Massal (CSV)"
])

# ==================== TAB 1: DASHBOARD ====================
with tab_dashboard:
    if metrics_data:
        # Metrik KPI Utama
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        with col_kpi1:
            st.metric("Total Sampel Dataset", "5.000 Mahasiswa")
        with col_kpi2:
            st.metric("Akurasi Model Kelulusan", f"{(metrics_data['model_graduation']['accuracy'] * 100):.2f}%")
        with col_kpi3:
            st.metric("Akurasi Model Karier", f"{(metrics_data['model_career']['accuracy'] * 100):.2f}%")
            
        st.write("---")
        
        # Baris Grafik 1
        col_g1, col_g2 = st.columns([3, 2])
        
        with col_g1:
            st.subheader("🔑 Fitur Paling Berpengaruh Terhadap Kelulusan")
            # Plot Feature Importance
            feat_imp = metrics_data['model_graduation']['feature_importance']
            df_feat = pd.DataFrame(feat_imp, columns=['Fitur', 'Skor Pengaruh']).sort_values('Skor Pengaruh', ascending=True)
            
            # Ganti nama label agar bahasa indonesia
            df_feat['Fitur'] = df_feat['Fitur'].replace({
                'IPK': 'IPK Kumulatif',
                'Persentase_Kehadiran': 'Kehadiran (%)',
                'SKS_Lulus': 'SKS Lulus',
                'IPS_Terakhir': 'IPS Terakhir',
                'Skor_Softskill': 'Skor Softskill',
                'Pengalaman_Magang_Ya': 'Magang: Ya',
                'Pengalaman_Magang_Tidak': 'Magang: Tidak',
                'Jumlah_Sertifikasi': 'Jumlah Sertifikasi'
            })
            
            fig, ax = plt.subplots(figsize=(8, 4.5))
            fig.patch.set_facecolor('#0b0f19')
            ax.set_facecolor('#111827')
            
            bars = ax.barh(df_feat['Fitur'], df_feat['Skor Pengaruh'], color='#6366f1', edgecolor='#4f46e5', height=0.6)
            ax.tick_params(colors='#9ca3af', labelsize=9)
            ax.spines['bottom'].set_color('#252f3f')
            ax.spines['left'].set_color('#252f3f')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.xaxis.grid(True, color='rgba(255,255,255,0.05)')
            ax.set_xlabel('Skor Kepentingan Relatif', color='#9ca3af', fontsize=9)
            
            st.pyplot(fig)
            
        with col_g2:
            st.subheader("🍰 Distribusi Kelulusan Historis")
            # Doughnut chart kelulusan
            grad_dist = metrics_data['eda']['grad_distribution']
            
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.patch.set_facecolor('#0b0f19')
            
            wedges, texts, autotexts = ax.pie(
                grad_dist.values(), 
                labels=grad_dist.keys(), 
                autopct='%1.1f%%',
                startangle=90, 
                colors=['#10b981', '#ef4444'],
                textprops=dict(color='#9ca3af', fontsize=10),
                wedgeprops=dict(width=0.4, edgecolor='#0b0f19')
            )
            for autotext in autotexts:
                autotext.set_color('#ffffff')
                autotext.set_weight('bold')
                
            st.pyplot(fig)
            
        st.write("---")
        
        # Baris Grafik 2
        col_g3, col_g4 = st.columns(2)
        
        with col_g3:
            st.subheader("📈 Hubungan Rata-rata IPK & Kelulusan")
            ipk_data = metrics_data['eda']['ipk_by_grad']
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig.patch.set_facecolor('#0b0f19')
            ax.set_facecolor('#111827')
            
            ax.bar(ipk_data.keys(), ipk_data.values(), color=['#10b981', '#ef4444'], width=0.4)
            ax.set_ylim(2.0, 4.0)
            ax.tick_params(colors='#9ca3af')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#252f3f')
            ax.spines['left'].set_color('#252f3f')
            ax.set_ylabel('IPK Rata-rata', color='#9ca3af')
            ax.grid(axis='y', color='rgba(255,255,255,0.05)')
            
            st.pyplot(fig)
            
        with col_g4:
            st.subheader("💼 Dampak Magang Pada Kelulusan")
            magang_data = metrics_data['eda']['magang_vs_grad']
            categories = list(magang_data.keys()) # ['Tidak', 'Ya']
            tepat_vals = [magang_data[cat]['Tepat Waktu'] * 100 for cat in categories]
            lambat_vals = [magang_data[cat]['Tidak Tepat Waktu'] * 100 for cat in categories]
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig.patch.set_facecolor('#0b0f19')
            ax.set_facecolor('#111827')
            
            ax.bar(categories, tepat_vals, label='Tepat Waktu', color='#10b981', width=0.4)
            ax.bar(categories, lambat_vals, bottom=tepat_vals, label='Tidak Tepat Waktu', color='#ef4444', width=0.4)
            ax.tick_params(colors='#9ca3af')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#252f3f')
            ax.spines['left'].set_color('#252f3f')
            ax.set_ylabel('Persentase (%)', color='#9ca3af')
            ax.legend(facecolor='#111827', edgecolor='none', labelcolor='#9ca3af')
            
            st.pyplot(fig)
            
        # Baris Grafik 3: Sebaran Arah Karier
        st.write("---")
        st.subheader("🎯 Sebaran Pilihan Arah Karier Mahasiswa")
        career_dist = metrics_data['eda']['career_distribution']
        df_career = pd.DataFrame(list(career_dist.items()), columns=['Karier', 'Jumlah']).sort_values('Jumlah', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor('#0b0f19')
        ax.set_facecolor('#111827')
        
        ax.bar(df_career['Karier'], df_career['Jumlah'], color='rgba(139, 92, 246, 0.7)', edgecolor='#8b5cf6', width=0.6)
        ax.tick_params(colors='#9ca3af', axis='x', rotation=45)
        ax.tick_params(colors='#9ca3af', axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#252f3f')
        ax.spines['left'].set_color('#252f3f')
        ax.set_ylabel('Jumlah Mahasiswa', color='#9ca3af')
        ax.grid(axis='y', color='rgba(255,255,255,0.05)')
        
        st.pyplot(fig)
    else:
        st.warning("⚠️ Berkas `metrics.json` tidak ditemukan. Silakan jalankan `train_models.py` terlebih dahulu untuk melatih model dan menghasilkan statistik.")

# ==================== TAB 2: SIMULATOR ====================
with tab_single:
    if model_graduation is None or model_career is None:
        st.error("❌ Berkas model machine learning (`.joblib`) tidak ditemukan di server. Harap jalankan script pelatihan model terlebih dahulu.")
    else:
        # Layout Form dan Hasil berdampingan
        col_form, col_result = st.columns([1.2, 1])
        
        with col_form:
            st.markdown("### 📝 Input Profil Mahasiswa")
            
            with st.container():
                # Kategori A: Akademis
                st.markdown("<h4 style='color:#6366f1; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom:5px;'>A. Profil Akademik</h4>", unsafe_allow_html=True)
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    prodi = st.selectbox("Program Studi", ["Komputer", "Manajemen", "Bisnis Digital"])
                    ipk = st.number_input("IPK Kumulatif (0.0 - 4.0)", min_value=0.0, max_value=4.0, value=3.25, step=0.01)
                with col_a2:
                    ips_terakhir = st.number_input("IPS Terakhir", min_value=0.0, max_value=4.0, value=3.30, step=0.01)
                    kehadiran = st.slider("Persentase Kehadiran (%)", min_value=0, max_value=100, value=90)
                sks = st.number_input("Total SKS Lulus", min_value=0, max_value=160, value=120)
                
                # Kategori B: Kompetensi
                st.markdown("<br><h4 style='color:#6366f1; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom:5px;'>B. Portofolio & Kompetensi</h4>", unsafe_allow_html=True)
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    magang = st.selectbox("Pengalaman Magang", ["Ya", "Tidak"])
                    sertifikasi = st.number_input("Jumlah Sertifikasi", min_value=0, max_value=20, value=2)
                with col_b2:
                    organisasi = st.number_input("Jumlah Organisasi Diikuti", min_value=0, max_value=10, value=1)
                    softskill = st.slider("Skor Softskill (1-10)", min_value=1, max_value=10, value=7)
                sosial = st.slider("Skor Aktivitas Sosial (1-10)", min_value=1, max_value=10, value=6)
                
                # Kategori C: Pendukung Eksternal
                st.markdown("<br><h4 style='color:#6366f1; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom:5px;'>C. Sosial & Pendukung Eksternal</h4>", unsafe_allow_html=True)
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    ekonomi = st.selectbox("Kondisi Ekonomi Keluarga", ["Menengah", "Tinggi", "Rendah"])
                    internet = st.selectbox("Kualitas Akses Internet", ["Baik", "Cukup", "Kurang"])
                with col_c2:
                    pekerjaan = st.selectbox("Status Pekerjaan Mahasiswa", ["Tidak Bekerja", "Part-time", "Freelance"])
                    keluarga = st.slider("Skor Dukungan Keluarga (1-5)", min_value=1, max_value=5, value=4)
                    
            st.write(" ")
            btn_predict = st.button("🔮 Jalankan Prediksi Data Mining", use_container_width=True)
            
        with col_result:
            st.markdown("### 📊 Hasil Prediksi Evaluasi")
            
            if btn_predict:
                # Bentuk DataFrame input
                df_input = pd.DataFrame({
                    'Prodi': [prodi],
                    'IPK': [float(ipk)],
                    'IPS_Terakhir': [float(ips_terakhir)],
                    'Persentase_Kehadiran': [int(kehadiran)],
                    'SKS_Lulus': [int(sks)],
                    'Jumlah_Organisasi': [int(organisasi)],
                    'Kondisi_Ekonomi': [ekonomi],
                    'Kualitas_Akses_Internet': [internet],
                    'Pengalaman_Magang': [magang],
                    'Jumlah_Sertifikasi': [int(sertifikasi)],
                    'Aktivitas_Sosial_Skor': [int(sosial)],
                    'Status_Pekerjaan': [pekerjaan],
                    'Dukungan_Keluarga': [int(keluarga)],
                    'Skor_Softskill': [int(softskill)]
                })
                
                # 1. Prediksi Kelulusan
                pred_grad = model_graduation.predict(df_input)[0]
                proba_grad_classes = model_graduation.classes_.tolist()
                proba_grad_vals = model_graduation.predict_proba(df_input)[0].tolist()
                idx_tepat = proba_grad_classes.index('Tepat Waktu') if 'Tepat Waktu' in proba_grad_classes else 0
                prob_tepat = proba_grad_vals[idx_tepat]
                
                # 2. Prediksi Karier (Top 3)
                proba_career_classes = model_career.classes_.tolist()
                proba_career_vals = model_career.predict_proba(df_input)[0].tolist()
                
                career_matches = sorted(
                    zip(proba_career_classes, proba_career_vals),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Render UI Hasil Prediksi
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                
                # Header hasil
                st.write("#### Kelulusan Tepat Waktu")
                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    # Tampilkan probabilitas melingkar sederhana
                    st.metric("Peluang Tepat Waktu", f"{prob_tepat*100:.1f}%")
                with col_res2:
                    if pred_grad == 'Tepat Waktu':
                        st.markdown("<span style='background-color:#10b981; color:white; padding:0.4rem 1rem; border-radius:50px; font-weight:bold;'>TEPAT WAKTU</span>", unsafe_allow_html=True)
                        st.write("Mahasiswa diprediksi lulus sesuai jadwal akademik reguler.")
                    else:
                        st.markdown("<span style='background-color:#ef4444; color:white; padding:0.4rem 1rem; border-radius:50px; font-weight:bold;'>RISIKO TERLAMBAT</span>", unsafe_allow_html=True)
                        st.write("Mahasiswa diprediksi mengalami hambatan kelulusan tepat waktu.")
                
                st.markdown("---")
                
                # Tampilkan Karier Terbaik
                st.write("#### 🎯 Arah Rekomendasi Karier Utama")
                top_career = career_matches[0]
                st.markdown(f"### **{top_career[0]}**")
                st.progress(float(top_career[1]))
                st.markdown(f"<small style='color:#6366f1;'>Kesesuaian Profil: <b>{top_career[1]*100:.1f}% Match</b></small>", unsafe_allow_html=True)
                
                # Alternatif karier
                st.write(" ")
                st.write("##### Karier Alternatif Lain:")
                for alt_c, alt_prob in career_matches[1:3]:
                    st.markdown(f"- **{alt_c}** ({alt_prob*100:.1f}% Match)")
                    
                st.markdown("---")
                
                # Tampilkan Rekomendasi Pintar
                st.write("#### 💡 Rekomendasi Taktis Tindakan")
                
                if pred_grad == 'Tidak Tepat Waktu' or prob_tepat < 0.6:
                    st.markdown(f"<div class='rec-box rec-danger'>🚨 <b>Peringatan Risiko Terlambat Lulus</b> (Peluang Lulus Tepat Waktu hanya {prob_tepat*100:.1f}%). Segera terapkan tindakan perbaikan berikut.</div>", unsafe_allow_html=True)
                    
                    if kehadiran < 80:
                        st.markdown(f"<div class='rec-box rec-warning'>⚠️ Kehadiran kelas saat ini ({kehadiran}%) berada di bawah ambang batas aman 80%. Prioritaskan absensi kehadiran kuliah.</div>", unsafe_allow_html=True)
                    
                    if ipk < 3.0:
                        st.markdown(f"<div class='rec-box rec-warning'>⚠️ Nilai IPK ({ipk}) kurang memuaskan. Targetkan bimbingan intensif dan tingkatkan waktu belajar mandiri untuk mendongkrak IPK.</div>", unsafe_allow_html=True)
                        
                    if sks < 110:
                        st.markdown(f"<div class='rec-box rec-warning'>⚠️ Beban SKS lulus ({sks} SKS) masih rendah. Konsultasikan dengan dosen wali untuk mengambil beban SKS maksimal di semester berikutnya.</div>", unsafe_allow_html=True)
                        
                    if softskill < 6:
                        st.markdown(f"<div class='rec-box rec-info'>💡 Tingkatkan kompetensi softskill Anda (Skor saat ini: {softskill}/10). Ikuti kursus publik speaking, kepemimpinan, atau organisasi kemahasiswaan.</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='rec-box rec-success'>✅ <b>Profil Mahasiswa Sangat Baik!</b> Peluang lulus tepat waktu sangat tinggi ({prob_tepat*100:.1f}%). Pertahankan konsistensi performa Anda.</div>", unsafe_allow_html=True)
                    
                    if magang == 'Tidak':
                        st.markdown("<div class='rec-box rec-info'>💡 Untuk mempersiapkan transisi karier ke arah <b>" + top_career[0] + "</b>, mulailah melamar program magang di semester berjalan ini.</div>", unsafe_allow_html=True)
                        
                    if sertifikasi < 2:
                        st.markdown(f"<div class='rec-box rec-info'>💡 Tambah minimal 1 sertifikasi profesi kompetensi tambahan untuk memperkuat portofolio karier Anda (saat ini baru {sertifikasi} sertifikasi).</div>", unsafe_allow_html=True)
                        
                    if organisasi == 0:
                        st.markdown("<div class='rec-box rec-info'>💡 Bergabunglah dengan organisasi/unit kegiatan mahasiswa minimal 1 untuk memperluas networking sosial dan softskill kolaborasi.</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='text-align: center; padding: 5rem 2rem; color: #9ca3af;'>
                    <i class="fa-solid fa-gears" style="font-size: 3rem; opacity:0.3; margin-bottom:1rem; display:block;"></i>
                    <h4>Menunggu Input Profil</h4>
                    <p style="font-size:0.85rem;">Isi formulir parameter profil mahasiswa di sebelah kiri, lalu klik tombol 'Jalankan Prediksi' untuk melihat visualisasi laporan hasil evaluasi.</p>
                </div>
                """, unsafe_allow_html=True)

# ==================== TAB 3: BATCH PREDICT ====================
with tab_batch:
    st.markdown("### 📂 Pengolahan Prediksi CSV Secara Massal")
    st.write("Gunakan fitur ini untuk memproses ratusan atau ribuan data mahasiswa sekaligus dengan mengunggah daftar mahasiswa berformat CSV.")
    
    # Panduan schema
    with st.expander("ℹ️ Panduan Format Struktur Kolom CSV"):
        st.write("Berkas CSV Anda harus memiliki kolom dengan nama-nama persis berikut (case-sensitive):")
        st.code("Prodi, IPK, IPS_Terakhir, Persentase_Kehadiran, SKS_Lulus, Jumlah_Organisasi, Kondisi_Ekonomi, Kualitas_Akses_Internet, Pengalaman_Magang, Jumlah_Sertifikasi, Aktivitas_Sosial_Skor, Status_Pekerjaan, Dukungan_Keluarga, Skor_Softskill")
        
    uploaded_file = st.file_uploader("Unggah Berkas CSV Mahasiswa", type="csv")
    
    if uploaded_file is not None:
        try:
            # Membaca data
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"✔️ Berkas '{uploaded_file.name}' berhasil diunggah.")
            
            st.write("#### 🔍 Pratinjau 5 Data Pertama Terunggah:")
            st.dataframe(df_batch.head(5), use_container_width=True)
            
            # Cek kolom wajib
            required_cols = [
                'Prodi', 'IPK', 'IPS_Terakhir', 'Persentase_Kehadiran', 'SKS_Lulus',
                'Jumlah_Organisasi', 'Kondisi_Ekonomi', 'Kualitas_Akses_Internet',
                'Pengalaman_Magang', 'Jumlah_Sertifikasi', 'Aktivitas_Sosial_Skor',
                'Status_Pekerjaan', 'Dukungan_Keluarga', 'Skor_Softskill'
            ]
            
            missing_cols = [c for c in required_cols if c not in df_batch.columns]
            
            if missing_cols:
                st.error(f"❌ Struktur CSV tidak valid! Kolom berikut tidak ditemukan: {', '.join(missing_cols)}")
            else:
                st.write(" ")
                btn_process_batch = st.button("🚀 Jalankan Proses Analisis Data Mining Massal", use_container_width=True)
                
                if btn_process_batch:
                    with st.spinner("Model Random Forest sedang memproses data..."):
                        # Siapkan subset fitur untuk prediksi
                        X_batch = df_batch[required_cols].copy()
                        
                        # 1. Prediksi Kelulusan
                        pred_grad = model_graduation.predict(X_batch)
                        proba_grad = model_graduation.predict_proba(X_batch)
                        grad_classes = model_graduation.classes_.tolist()
                        idx_tepat = grad_classes.index('Tepat Waktu') if 'Tepat Waktu' in grad_classes else 0
                        prob_tepat = proba_grad[:, idx_tepat]
                        
                        # 2. Prediksi Karier
                        pred_career = model_career.predict(X_batch)
                        proba_career = model_career.predict_proba(X_batch)
                        career_classes = model_career.classes_.tolist()
                        
                        # Tambahkan kolom prediksi ke DF asli
                        df_batch['Prediksi_Kelulusan'] = pred_grad
                        df_batch['Probabilitas_Tepat_Waktu_Persen'] = np.round(prob_tepat * 100, 1)
                        df_batch['Rekomendasi_Karier_Utama'] = pred_career
                        
                        # Ambil alternatif karier ke-2
                        second_careers = []
                        for idx_row in range(len(proba_career)):
                            row_probs = proba_career[idx_row]
                            sorted_idx = np.argsort(row_probs)[::-1]
                            alt_idx = sorted_idx[1] if len(sorted_idx) > 1 else sorted_idx[0]
                            second_careers.append(career_classes[alt_idx])
                            
                        df_batch['Rekomendasi_Karier_Alternatif'] = second_careers
                        
                        # Hitung Ringkasan hasil
                        total_rows = len(df_batch)
                        tepat_count = sum(df_batch['Prediksi_Kelulusan'] == 'Tepat Waktu')
                        lambat_count = total_rows - tepat_count
                        
                        # Tampilkan Statistik Batch
                        st.markdown("---")
                        st.write("#### 📊 Hasil Analisis Prediktif")
                        col_b1, col_b2, col_b3 = st.columns(3)
                        with col_b1:
                            st.metric("Total Baris Diproses", f"{total_rows} Mahasiswa")
                        with col_b2:
                            st.metric("Diprediksi Tepat Waktu", f"{tepat_count} Mahasiswa", delta=f"{(tepat_count/total_rows*100):.1f}%")
                        with col_b3:
                            st.metric("Risiko Terlambat Lulus", f"{lambat_count} Mahasiswa", delta=f"-{(lambat_count/total_rows*100):.1f}%", delta_color="inverse")
                            
                        st.write(" ")
                        st.write("#### 📋 Hasil Pratinjau 5 Baris Pertama Terprediksi:")
                        preview_cols = ['Prodi', 'IPK', 'Persentase_Kehadiran', 'Prediksi_Kelulusan', 'Probabilitas_Tepat_Waktu_Persen', 'Rekomendasi_Karier_Utama']
                        st.dataframe(df_batch[preview_cols].head(5), use_container_width=True)
                        
                        # Tombol Download
                        csv_data = df_batch.to_csv(index=False).encode('utf-8')
                        st.write("---")
                        st.download_button(
                            label="📥 Unduh File CSV Hasil Analisis Kelulusan & Karier",
                            data=csv_data,
                            file_name=f"hasil_analisis_{uploaded_file.name}",
                            mime="text/csv",
                            use_container_width=True
                        )
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan dalam pemrosesan berkas: {str(e)}")
