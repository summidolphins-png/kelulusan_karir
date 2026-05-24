import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

def main():
    print("Memulai proses analisis data dan pelatihan model...")
    
    # Path dataset
    dataset_path = "dataset_prediksi_mahasiswa_5000_final.csv"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} tidak ditemukan di workspace.")
        return
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    print(f"Dataset berhasil dimuat. Ukuran: {df.shape[0]} baris, {df.shape[1]} kolom.")
    
    # Definisikan kolom fitur
    numerical_features = [
        'IPK', 'IPS_Terakhir', 'Persentase_Kehadiran', 'SKS_Lulus', 
        'Jumlah_Organisasi', 'Jumlah_Sertifikasi', 'Aktivitas_Sosial_Skor', 
        'Dukungan_Keluarga', 'Skor_Softskill'
    ]
    
    categorical_features = [
        'Prodi', 'Kondisi_Ekonomi', 'Kualitas_Akses_Internet', 
        'Pengalaman_Magang', 'Status_Pekerjaan'
    ]
    
    X = df[numerical_features + categorical_features]
    
    # Target label
    y_graduation = df['Kelulusan_Tepat_Waktu']
    y_career = df['Arah_Karier']
    
    # Pra-pemrosesan data menggunakan Scikit-Learn Pipeline
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Pemisahan data latih & data uji (80% latih, 20% uji)
    # 1. Kelulusan
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
        X, y_graduation, test_size=0.2, random_state=42, stratify=y_graduation
    )
    
    # 2. Arah Karier
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_career, test_size=0.2, random_state=42, stratify=y_career
    )
    
    print("\nMelatih Model 1: Klasifikasi Kelulusan Tepat Waktu...")
    # Pipeline Model Kelulusan (Random Forest)
    model_graduation = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    model_graduation.fit(X_train_g, y_train_g)
    y_pred_g = model_graduation.predict(X_test_g)
    acc_g = accuracy_score(y_test_g, y_pred_g)
    print(f"Akurasi Model Kelulusan: {acc_g * 100:.2f}%")
    
    # Evaluasi Detail Model Kelulusan
    cm_g = confusion_matrix(y_test_g, y_pred_g)
    report_g = classification_report(y_test_g, y_pred_g, output_dict=True)
    
    print("\nMelatih Model 2: Klasifikasi Arah Karier...")
    # Pipeline Model Arah Karier (Random Forest)
    model_career = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    model_career.fit(X_train_c, y_train_c)
    y_pred_c = model_career.predict(X_test_c)
    acc_c = accuracy_score(y_test_c, y_pred_c)
    print(f"Akurasi Model Arah Karier: {acc_c * 100:.2f}%")
    
    # Evaluasi Detail Model Arah Karier
    cm_c = confusion_matrix(y_test_c, y_pred_c)
    report_c = classification_report(y_test_c, y_pred_c, output_dict=True)
    
    # Menyimpan model latih
    joblib.dump(model_graduation, "model_graduation.joblib")
    joblib.dump(model_career, "model_career.joblib")
    print("\nModel model_graduation.joblib dan model_career.joblib berhasil disimpan!")
    
    # Mengekstrak nama fitur setelah encoding untuk Feature Importance
    # Mendapatkan nama kategori dari OneHotEncoder
    ohe = model_graduation.named_steps['preprocessor'].named_transformers_['cat']
    cat_encoder_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    feature_names = numerical_features + cat_encoder_feature_names
    
    # Feature Importance Model Kelulusan
    importances_g = model_graduation.named_steps['classifier'].feature_importances_
    feat_imp_g = sorted(zip(feature_names, importances_g.tolist()), key=lambda x: x[1], reverse=True)
    
    # Feature Importance Model Arah Karier
    importances_c = model_career.named_steps['classifier'].feature_importances_
    feat_imp_c = sorted(zip(feature_names, importances_c.tolist()), key=lambda x: x[1], reverse=True)
    
    # Mengumpulkan statistik dataset asli untuk EDA Dashboard
    print("\nMengekstrak statistik EDA untuk Dashboard...")
    
    # 1. Rata-rata IPK berdasarkan Kelulusan
    ipk_by_grad = df.groupby('Kelulusan_Tepat_Waktu')['IPK'].mean().to_dict()
    
    # 2. Rata-rata Kehadiran berdasarkan Kelulusan
    kehadiran_by_grad = df.groupby('Kelulusan_Tepat_Waktu')['Persentase_Kehadiran'].mean().to_dict()
    
    # 3. Pengaruh Magang terhadap Kelulusan (Cross-Tabulation)
    magang_vs_grad = pd.crosstab(df['Pengalaman_Magang'], df['Kelulusan_Tepat_Waktu'], normalize='index').to_dict('index')
    
    # 4. Distribusi Arah Karier per Prodi (Cross-Tabulation)
    prodi_vs_career = pd.crosstab(df['Prodi'], df['Arah_Karier']).to_dict('index')
    
    # 5. Rata-rata Skor Softskill per Arah Karier
    softskill_by_career = df.groupby('Arah_Karier')['Skor_Softskill'].mean().to_dict()
    
    # 6. Distribusi Kelulusan Tepat Waktu
    grad_distribution = df['Kelulusan_Tepat_Waktu'].value_counts().to_dict()
    
    # 7. Distribusi Arah Karier
    career_distribution = df['Arah_Karier'].value_counts().to_dict()
    
    # Gabungkan semua metrik dan statistik
    metrics_json = {
        'model_graduation': {
            'accuracy': float(acc_g),
            'confusion_matrix': cm_g.tolist(),
            'classes': model_graduation.classes_.tolist(),
            'classification_report': report_g,
            'feature_importance': feat_imp_g[:15] # Ambil 15 fitur paling berpengaruh
        },
        'model_career': {
            'accuracy': float(acc_c),
            'confusion_matrix': cm_c.tolist(),
            'classes': model_career.classes_.tolist(),
            'classification_report': report_c,
            'feature_importance': feat_imp_c[:15]
        },
        'eda': {
            'ipk_by_grad': ipk_by_grad,
            'kehadiran_by_grad': kehadiran_by_grad,
            'magang_vs_grad': magang_vs_grad,
            'prodi_vs_career': prodi_vs_career,
            'softskill_by_career': softskill_by_career,
            'grad_distribution': grad_distribution,
            'career_distribution': career_distribution
        }
    }
    
    # Simpan statistik ke metrics.json
    with open("metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=4)
        
    print("Metrik evaluasi dan statistik EDA disimpan di metrics.json!")
    print("Pelatihan model dan analisis data mining selesai sepenuhnya!")

if __name__ == "__main__":
    main()
