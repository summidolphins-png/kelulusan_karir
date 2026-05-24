import os
import json
import pandas as pd
import numpy as np
import joblib
from flask import Flask, jsonify, request, render_template, send_from_directory, Response

app = Flask(__name__, static_folder='static', template_folder='templates')

# Cek keberadaan model dan metrik
MODEL_GRADUATION_PATH = "model_graduation.joblib"
MODEL_CAREER_PATH = "model_career.joblib"
METRICS_PATH = "metrics.json"

model_graduation = None
model_career = None
metrics_data = None

def load_resources():
    global model_graduation, model_career, metrics_data
    if os.path.exists(MODEL_GRADUATION_PATH):
        model_graduation = joblib.load(MODEL_GRADUATION_PATH)
        print("Model kelulusan berhasil dimuat.")
    else:
        print("PERINGATAN: model_graduation.joblib tidak ditemukan!")

    if os.path.exists(MODEL_CAREER_PATH):
        model_career = joblib.load(MODEL_CAREER_PATH)
        print("Model arah karier berhasil dimuat.")
    else:
        print("PERINGATAN: model_career.joblib tidak ditemukan!")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics_data = json.load(f)
        print("Metrik & statistik EDA berhasil dimuat.")
    else:
        print("PERINGATAN: metrics.json tidak ditemukan!")

# Muat model saat server dinyalakan
load_resources()

@app.route('/')
def home():
    # Render antarmuka dashboard utama
    return render_template('index.html')

@app.route('/api/stats', methods=['GET'])
def get_stats():
    # Mengembalikan data statistik EDA dan metrik performa model untuk chart dashboard
    if metrics_data is None:
        # Jika belum dimuat, coba muat ulang
        load_resources()
    
    if metrics_data:
        return jsonify(metrics_data)
    else:
        return jsonify({"error": "Data statistik tidak tersedia"}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    if model_graduation is None or model_career is None:
        return jsonify({"error": "Model machine learning belum siap"}), 500
        
    try:
        data = request.get_json()
        
        # Validasi field masukan
        required_fields = [
            'Prodi', 'IPK', 'IPS_Terakhir', 'Persentase_Kehadiran', 'SKS_Lulus',
            'Jumlah_Organisasi', 'Kondisi_Ekonomi', 'Kualitas_Akses_Internet',
            'Pengalaman_Magang', 'Jumlah_Sertifikasi', 'Aktivitas_Sosial_Skor',
            'Status_Pekerjaan', 'Dukungan_Keluarga', 'Skor_Softskill'
        ]
        
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Kolom input tidak lengkap: {', '.join(missing)}"}), 400
            
        # Bentuk DataFrame satu baris untuk prediksi
        input_data = {
            'Prodi': [str(data['Prodi'])],
            'IPK': [float(data['IPK'])],
            'IPS_Terakhir': [float(data['IPS_Terakhir'])],
            'Persentase_Kehadiran': [int(data['Persentase_Kehadiran'])],
            'SKS_Lulus': [int(data['SKS_Lulus'])],
            'Jumlah_Organisasi': [int(data['Jumlah_Organisasi'])],
            'Kondisi_Ekonomi': [str(data['Kondisi_Ekonomi'])],
            'Kualitas_Akses_Internet': [str(data['Kualitas_Akses_Internet'])],
            'Pengalaman_Magang': [str(data['Pengalaman_Magang'])],
            'Jumlah_Sertifikasi': [int(data['Jumlah_Sertifikasi'])],
            'Aktivitas_Sosial_Skor': [int(data['Aktivitas_Sosial_Skor'])],
            'Status_Pekerjaan': [str(data['Status_Pekerjaan'])],
            'Dukungan_Keluarga': [int(data['Dukungan_Keluarga'])],
            'Skor_Softskill': [int(data['Skor_Softskill'])]
        }
        
        df_input = pd.DataFrame(input_data)
        
        # 1. Prediksi Kelulusan Tepat Waktu
        pred_grad = model_graduation.predict(df_input)[0]
        proba_grad_classes = model_graduation.classes_.tolist()
        proba_grad_vals = model_graduation.predict_proba(df_input)[0].tolist()
        
        # Cari indeks untuk kelas 'Tepat Waktu'
        idx_tepat = proba_grad_classes.index('Tepat Waktu') if 'Tepat Waktu' in proba_grad_classes else 0
        prob_tepat_waktu = proba_grad_vals[idx_tepat]
        
        # 2. Prediksi Arah Karier (Rekomendasikan top 3)
        proba_career_classes = model_career.classes_.tolist()
        proba_career_vals = model_career.predict_proba(df_input)[0].tolist()
        
        career_matches = sorted(
            zip(proba_career_classes, proba_career_vals), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        top_careers = []
        for career, prob in career_matches[:3]:
            top_careers.append({
                "career": career,
                "score": round(prob * 100, 1)
            })
            
        # 3. Logika Rekomendasi Pintar Eksklusif
        recommendations = []
        ipk_val = float(data['IPK'])
        kehadiran_val = int(data['Persentase_Kehadiran'])
        sks_val = int(data['SKS_Lulus'])
        softskill_val = int(data['Skor_Softskill'])
        magang_val = str(data['Pengalaman_Magang'])
        sertif_val = int(data['Jumlah_Sertifikasi'])
        org_val = int(data['Jumlah_Organisasi'])
        
        if pred_grad == 'Tidak Tepat Waktu' or prob_tepat_waktu < 0.6:
            recommendations.append({
                "type": "danger",
                "text": f"Risiko keterlambatan lulus terdeteksi (Probabilitas Tepat Waktu hanya {prob_tepat_waktu*100:.1f}%)."
            })
            
            if kehadiran_val < 80:
                recommendations.append({
                    "type": "warning",
                    "text": f"Tingkatkan kehadiran kelas Anda yang saat ini {kehadiran_val}%. Kehadiran di bawah 80% merupakan faktor risiko utama kelulusan terlambat."
                })
            
            if ipk_val < 3.0:
                recommendations.append({
                    "type": "warning",
                    "text": f"Kinerja akademik perlu ditingkatkan. IPK Anda ({ipk_val}) berada di bawah standar aman 3.0. Buat jadwal belajar mandiri atau ikuti kelompok belajar."
                })
                
            if sks_val < 110:
                recommendations.append({
                    "type": "warning",
                    "text": f"Jumlah SKS lulus Anda ({sks_val}) masih rendah. Pastikan Anda mengambil beban SKS maksimal di semester berjalan untuk mengejar ketertinggalan."
                })
                
            if softskill_val < 6:
                recommendations.append({
                    "type": "info",
                    "text": f"Asah kompetensi softskill Anda (skor saat ini: {softskill_val}/10). Kurangnya softskill membatasi daya saing kerja dan kemampuan kolaborasi akademik."
                })
        else:
            recommendations.append({
                "type": "success",
                "text": f"Luar biasa! Profil Anda sangat solid dengan peluang kelulusan tepat waktu mencapai {prob_tepat_waktu*100:.1f}%."
            })
            
            if magang_val == 'Tidak':
                recommendations.append({
                    "type": "info",
                    "text": "Untuk meningkatkan daya saing karier yang direkomendasikan, cobalah mendaftar magang (internship) pada semester ini."
                })
                
            if sertif_val < 2:
                recommendations.append({
                    "type": "info",
                    "text": f"Anda baru memiliki {sertif_val} sertifikasi. Tambah minimal 1 sertifikasi profesional untuk memperkuat kesesuaian dengan arah karier {top_careers[0]['career']}."
                })
                
            if org_val == 0:
                recommendations.append({
                    "type": "info",
                    "text": "Pertimbangkan untuk bergabung dengan minimal 1 organisasi kemahasiswaan untuk memperluas jaringan sosial dan mengasah kerja sama tim."
                })

        return jsonify({
            "status": "success",
            "prediction": {
                "graduation": pred_grad,
                "graduation_probability": round(prob_tepat_waktu * 100, 1),
                "top_careers": top_careers,
                "recommendations": recommendations
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Gagal memproses prediksi: {str(e)}"}), 500

@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    if model_graduation is None or model_career is None:
        return jsonify({"error": "Model machine learning belum siap"}), 500
        
    try:
        # Cek file upload
        if 'file' not in request.files:
            return jsonify({"error": "Tidak ada file yang diunggah"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Nama file kosong"}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Format berkas harus berupa CSV"}), 400
            
        # Membaca CSV
        df = pd.read_csv(file)
        
        # Validasi kolom yang dibutuhkan
        required_features = [
            'Prodi', 'IPK', 'IPS_Terakhir', 'Persentase_Kehadiran', 'SKS_Lulus',
            'Jumlah_Organisasi', 'Kondisi_Ekonomi', 'Kualitas_Akses_Internet',
            'Pengalaman_Magang', 'Jumlah_Sertifikasi', 'Aktivitas_Sosial_Skor',
            'Status_Pekerjaan', 'Dukungan_Keluarga', 'Skor_Softskill'
        ]
        
        missing = [col for col in required_features if col not in df.columns]
        if missing:
            return jsonify({"error": f"Kolom berikut tidak ditemukan di CSV: {', '.join(missing)}"}), 400
            
        # Salin data untuk prediksi
        X_batch = df[required_features].copy()
        
        # Lakukan prediksi kelulusan
        pred_grad = model_graduation.predict(X_batch)
        proba_grad = model_graduation.predict_proba(X_batch)
        grad_classes = model_graduation.classes_.tolist()
        idx_tepat = grad_classes.index('Tepat Waktu') if 'Tepat Waktu' in grad_classes else 0
        prob_tepat = proba_grad[:, idx_tepat]
        
        # Lakukan prediksi karier
        pred_career = model_career.predict(X_batch)
        proba_career = model_career.predict_proba(X_batch)
        career_classes = model_career.classes_.tolist()
        
        # Tambahkan kolom hasil ke DataFrame
        df['Prediksi_Kelulusan'] = pred_grad
        df['Probabilitas_Tepat_Waktu_Persen'] = np.round(prob_tepat * 100, 1)
        df['Rekomendasi_Karier_Utama'] = pred_career
        
        # Tambahkan rekomendasi karier ke-2 (alternatif)
        second_careers = []
        for i in range(len(proba_career)):
            row_probs = proba_career[i]
            sorted_idx = np.argsort(row_probs)[::-1]
            # Ambil indeks ke-2 teratas (jika ada lebih dari 1 kelas)
            alt_idx = sorted_idx[1] if len(sorted_idx) > 1 else sorted_idx[0]
            second_careers.append(career_classes[alt_idx])
            
        df['Rekomendasi_Karier_Alternatif'] = second_careers
        
        # Ubah ke CSV untuk diunduh kembali
        output_csv = df.to_csv(index=False)
        
        filename = "prediksi_" + file.filename
        
        return Response(
            output_csv,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        return jsonify({"error": f"Gagal memproses batch prediksi: {str(e)}"}), 500

if __name__ == '__main__':
    print("Menjalankan aplikasi Flask di http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
