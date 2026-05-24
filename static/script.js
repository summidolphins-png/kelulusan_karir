document.addEventListener("DOMContentLoaded", () => {
    
    // --- STATE VARIABLES ---
    let statsData = null;
    let batchBlob = null;
    let batchFilename = "hasil_prediksi_mahasiswa.csv";
    
    // Keep track of ChartJS instances to destroy them before rendering new ones
    const chartInstances = {};

    // --- TAB NAVIGATION SYSTEM ---
    const menuItems = document.querySelectorAll(".menu-item");
    const tabPanes = document.querySelectorAll(".tab-pane");
    const pageTitle = document.getElementById("page-title");
    const pageDesc = document.getElementById("page-desc");

    const tabMeta = {
        "dashboard": {
            title: "Dashboard Analisis & Data Mining",
            desc: "Visualisasi statistik dataset dan metrik performa model klasifikasi Random Forest."
        },
        "predict-single": {
            title: "Simulasi Evaluasi Mahasiswa Tunggal",
            desc: "Prediksi probabilitas kelulusan tepat waktu dan arah karier yang cocok untuk mahasiswa tertentu."
        },
        "predict-batch": {
            title: "Analisis Kelompok Secara Massal",
            desc: "Unggah daftar mahasiswa Anda dalam bentuk CSV untuk memproses prediksi berskala besar sekaligus."
        }
    };

    menuItems.forEach(item => {
        item.addEventListener("click", () => {
            const tabId = item.getAttribute("data-tab");
            
            // Toggle active menu button
            menuItems.forEach(i => i.classList.remove("active"));
            item.classList.add("active");

            // Toggle active tab pane
            tabPanes.forEach(pane => pane.classList.remove("active"));
            document.getElementById(`tab-${tabId}`).classList.add("active");

            // Update Header Meta
            if (tabMeta[tabId]) {
                pageTitle.textContent = tabMeta[tabId].title;
                pageDesc.textContent = tabMeta[tabId].desc;
            }
            
            // Re-render charts on dashboard navigation (responsive layout adjust)
            if (tabId === "dashboard") {
                renderAllCharts();
            }
        });
    });

    // --- SLIDER VALUE UPDATERS ---
    const setupSlider = (sliderId, valueId) => {
        const slider = document.getElementById(sliderId);
        const valueDisplay = document.getElementById(valueId);
        if (slider && valueDisplay) {
            slider.addEventListener("input", (e) => {
                valueDisplay.textContent = e.target.value;
            });
        }
    };
    
    setupSlider("Skor_Softskill", "val-softskill");
    setupSlider("Aktivitas_Sosial_Skor", "val-sosial");
    setupSlider("Dukungan_Keluarga", "val-keluarga");

    // --- CHART MANAGEMENT SYSTEM ---
    const destroyChart = (chartKey) => {
        if (chartInstances[chartKey]) {
            chartInstances[chartKey].destroy();
            delete chartInstances[chartKey];
        }
    };

    const renderAllCharts = () => {
        if (!statsData) return;

        const eda = statsData.eda;
        const modelGrad = statsData.model_graduation;
        const modelCareer = statsData.model_career;

        // Apply dark mode global styles to Chart.js
        Chart.defaults.color = '#9ca3af';
        Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
        Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.06)';

        // Update KPI card metrics
        document.getElementById("kpi-accuracy-grad").textContent = `${(modelGrad.accuracy * 100).toFixed(1)}%`;
        document.getElementById("kpi-accuracy-career").textContent = `${(modelCareer.accuracy * 100).toFixed(1)}%`;

        // 1. Chart Feature Importance (Horizontal Bar)
        destroyChart("featureGrad");
        const featLabels = modelGrad.feature_importance.map(item => {
            // Translate some features to Indonesian for better readability
            return item[0]
                .replace("IPK", "IPK Kumulatif")
                .replace("Persentase_Kehadiran", "Kehadiran (%)")
                .replace("SKS_Lulus", "SKS Lulus")
                .replace("IPS_Terakhir", "IPS Terakhir")
                .replace("Skor_Softskill", "Skor Softskill")
                .replace("Pengalaman_Magang_Ya", "Magang: Ya")
                .replace("Pengalaman_Magang_Tidak", "Magang: Tidak")
                .replace("Kondisi_Ekonomi_Rendah", "Ekonomi: Rendah")
                .replace("Kondisi_Ekonomi_Tinggi", "Ekonomi: Tinggi")
                .replace("Kondisi_Ekonomi_Menengah", "Ekonomi: Menengah")
                .replace("Jumlah_Sertifikasi", "Sertifikasi")
                .replace("Jumlah_Organisasi", "Organisasi")
                .replace("Aktivitas_Sosial_Skor", "Skor Sosial");
        });
        const featValues = modelGrad.feature_importance.map(item => item[1]);
        
        const ctxFeat = document.getElementById("chartFeatureGrad").getContext("2d");
        chartInstances["featureGrad"] = new Chart(ctxFeat, {
            type: 'bar',
            data: {
                labels: featLabels,
                datasets: [{
                    label: 'Skor Pengaruh',
                    data: featValues,
                    backgroundColor: 'rgba(99, 102, 241, 0.65)',
                    borderColor: '#6366f1',
                    borderWidth: 1,
                    borderRadius: 5,
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { grid: { display: false } }
                }
            }
        });

        // 2. Chart Graduation Distribution (Doughnut)
        destroyChart("distGrad");
        const gradLabels = Object.keys(eda.grad_distribution);
        const gradValues = Object.values(eda.grad_distribution);
        
        const ctxDist = document.getElementById("chartDistributionGrad").getContext("2d");
        chartInstances["distGrad"] = new Chart(ctxDist, {
            type: 'doughnut',
            data: {
                labels: gradLabels,
                datasets: [{
                    data: gradValues,
                    backgroundColor: ['#10b981', '#ef4444'],
                    borderColor: '#111827',
                    borderWidth: 3,
                    hoverOffset: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { boxWidth: 12, padding: 15 }
                    }
                },
                cutout: '65%'
            }
        });

        // 3. Chart IPK vs Graduation (Bar)
        destroyChart("ipkGrad");
        const ipkLabels = Object.keys(eda.ipk_by_grad);
        const ipkValues = Object.values(eda.ipk_by_grad);

        const ctxIpk = document.getElementById("chartIpkGrad").getContext("2d");
        chartInstances["ipkGrad"] = new Chart(ctxIpk, {
            type: 'bar',
            data: {
                labels: ipkLabels,
                datasets: [{
                    data: ipkValues,
                    backgroundColor: ['#10b981', '#ef4444'],
                    borderRadius: 8,
                    maxBarThickness: 50
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        min: 2.0,
                        max: 4.0,
                        title: { display: true, text: 'Rata-rata IPK' }
                    },
                    x: { grid: { display: false } }
                }
            }
        });

        // 4. Chart Internship Magang vs Graduation (Stacked Bar)
        destroyChart("magangGrad");
        const magangCategories = Object.keys(eda.magang_vs_grad); // ['Tidak', 'Ya']
        const tepatProbs = magangCategories.map(cat => eda.magang_vs_grad[cat]['Tepat Waktu'] * 100);
        const terlambatProbs = magangCategories.map(cat => eda.magang_vs_grad[cat]['Tidak Tepat Waktu'] * 100);

        const ctxMagang = document.getElementById("chartMagangGrad").getContext("2d");
        chartInstances["magangGrad"] = new Chart(ctxMagang, {
            type: 'bar',
            data: {
                labels: magangCategories.map(c => c === 'Ya' ? 'Pernah Magang' : 'Tidak Pernah Magang'),
                datasets: [
                    {
                        label: 'Tepat Waktu',
                        data: tepatProbs,
                        backgroundColor: '#10b981'
                    },
                    {
                        label: 'Tidak Tepat Waktu',
                        data: terlambatProbs,
                        backgroundColor: '#ef4444'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { stacked: true, grid: { display: false } },
                    y: { stacked: true, max: 100, title: { display: true, text: 'Persentase (%)' } }
                },
                plugins: {
                    legend: { position: 'bottom', labels: { boxWidth: 12 } }
                }
            }
        });

        // 5. Chart Career Distribution (Vertical Bar)
        destroyChart("careerDistribution");
        const careerLabels = Object.keys(eda.career_distribution);
        const careerValues = Object.values(eda.career_distribution);

        const ctxCareer = document.getElementById("chartCareerDistribution").getContext("2d");
        chartInstances["careerDistribution"] = new Chart(ctxCareer, {
            type: 'bar',
            data: {
                labels: careerLabels,
                datasets: [{
                    label: 'Jumlah Mahasiswa',
                    data: careerValues,
                    backgroundColor: 'rgba(139, 92, 246, 0.65)',
                    borderColor: '#8b5cf6',
                    borderWidth: 1,
                    borderRadius: 8,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { grid: { display: false }, ticks: { autoSkip: false, maxRotation: 45, minRotation: 45 } },
                    y: { title: { display: true, text: 'Frekuensi Mahasiswa' } }
                }
            }
        });
    };

    // Load initial stats
    const fetchStats = async () => {
        try {
            const res = await fetch("/api/stats");
            if (res.ok) {
                statsData = await res.json();
                renderAllCharts();
            } else {
                console.error("Gagal mengambil metrik model.");
            }
        } catch (err) {
            console.error("Error saat fetch metrik:", err);
        }
    };
    
    // Fetch stats automatically
    fetchStats();

    // --- TAB 2: SINGLE PREDICT SIMULATION HANDLER ---
    const singleForm = document.getElementById("single-predict-form");
    const resultPlaceholder = document.getElementById("result-placeholder");
    const resultContent = document.getElementById("result-content");
    const predictionResultPanel = document.getElementById("prediction-result-panel");

    singleForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        // 1. Show loading state in the result panel
        resultPlaceholder.classList.add("hidden");
        resultContent.classList.add("hidden");
        
        // Render a nice temporary spinner within the placeholder
        resultPlaceholder.innerHTML = `
            <div class="placeholder-icon"><i class="fa-solid fa-arrows-spin fa-spin-fast"></i></div>
            <h3>Sedang Menambang Data...</h3>
            <p>Model Random Forest sedang memproses kombinasi parameter akademik, sosial, dan ekonomi untuk menghitung probabilitas.</p>
        `;
        resultPlaceholder.classList.remove("hidden");

        // 2. Gather values from form
        const payload = {
            Prodi: document.getElementById("Prodi").value,
            IPK: parseFloat(document.getElementById("IPK").value),
            IPS_Terakhir: parseFloat(document.getElementById("IPS_Terakhir").value),
            Persentase_Kehadiran: parseInt(document.getElementById("Persentase_Kehadiran").value),
            SKS_Lulus: parseInt(document.getElementById("SKS_Lulus").value),
            Jumlah_Organisasi: parseInt(document.getElementById("Jumlah_Organisasi").value),
            Kondisi_Ekonomi: document.getElementById("Kondisi_Ekonomi").value,
            Kualitas_Akses_Internet: document.getElementById("Kualitas_Akses_Internet").value,
            Pengalaman_Magang: document.getElementById("Pengalaman_Magang").value,
            Jumlah_Sertifikasi: parseInt(document.getElementById("Jumlah_Sertifikasi").value),
            Aktivitas_Sosial_Skor: parseInt(document.getElementById("Aktivitas_Sosial_Skor").value),
            Status_Pekerjaan: document.getElementById("Status_Pekerjaan").value,
            Dukungan_Keluarga: parseInt(document.getElementById("Dukungan_Keluarga").value),
            Skor_Softskill: parseInt(document.getElementById("Skor_Softskill").value)
        };

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                const data = await res.json();
                
                // Hide loader
                resultPlaceholder.classList.add("hidden");
                resultContent.classList.remove("hidden");

                // Update result widgets
                const pred = data.prediction;
                
                // A. Graduation Gauge
                const prob = pred.graduation_probability;
                document.getElementById("grad-probability-percent").textContent = `${prob.toFixed(1)}%`;
                
                const gradCircle = document.querySelector(".grad-circle");
                const gradBadge = document.getElementById("grad-status-badge");
                const gradText = document.getElementById("grad-status-text");
                
                if (pred.graduation === "Tepat Waktu") {
                    gradCircle.style.background = `conic-gradient(#10b981 0% ${prob}%, rgba(255, 255, 255, 0.06) ${prob}% 100%)`;
                    gradCircle.style.boxShadow = "0 0 20px rgba(16, 185, 129, 0.25)";
                    gradBadge.textContent = "Tepat Waktu";
                    gradBadge.className = "status-badge status-tepat";
                    gradText.textContent = `Bagus! Mahasiswa diprediksi lulus sesuai waktu studi reguler 4 tahun (8 semester) dengan tingkat keyakinan ${prob.toFixed(1)}%.`;
                } else {
                    gradCircle.style.background = `conic-gradient(#ef4444 0% ${100 - prob}%, rgba(255, 255, 255, 0.06) ${100 - prob}% 100%)`;
                    gradCircle.style.boxShadow = "0 0 20px rgba(239, 68, 68, 0.25)";
                    gradBadge.textContent = "Risiko Terlambat";
                    gradBadge.className = "status-badge status-lambat";
                    gradText.textContent = `Peringatan: Mahasiswa memiliki risiko keterlambatan lulus dengan tingkat risiko ${(100 - prob).toFixed(1)}%. Segera lakukan pembinaan.`;
                }

                // B. Top Career matches
                const topCareer = pred.top_careers[0];
                document.getElementById("top-career-name").textContent = topCareer.career;
                document.getElementById("top-career-progress").style.width = `${topCareer.score}%`;
                document.getElementById("top-career-score").textContent = `${topCareer.score.toFixed(1)}% Match`;

                // Set career icon dynamically
                const careerIcons = {
                    "Software Developer": "fa-laptop-code",
                    "Data Analyst": "fa-chart-pie",
                    "Startup": "fa-rocket",
                    "Lanjut Studi": "fa-user-graduate",
                    "Digital Marketing": "fa-hashtag",
                    "Marketing": "fa-bullhorn",
                    "Human Resource": "fa-people-group",
                    "Wirausaha": "fa-store",
                    "Wirausaha Digital": "fa-globe",
                    "E-Commerce": "fa-cart-shopping",
                    "Pemerintahan": "fa-building-columns",
                    "Content Strategist": "fa-pen-nib",
                    "Industri Teknologi": "fa-microchip",
                    "Industri Manajemen": "fa-briefcase"
                };
                
                const cIcon = document.querySelector(".career-icon i");
                cIcon.className = `fa-solid ${careerIcons[topCareer.career] || 'fa-briefcase'}`;

                // Inject Alt careers
                const altContainer = document.getElementById("alt-careers-list");
                altContainer.innerHTML = "";
                
                pred.top_careers.slice(1).forEach(alt => {
                    const altItem = document.createElement("div");
                    altItem.className = "alt-career-item";
                    altItem.innerHTML = `
                        <span><i class="fa-solid ${careerIcons[alt.career] || 'fa-briefcase'} alt-icon" style="margin-right:8px; opacity:0.6;"></i> ${alt.career}</span>
                        <span class="alt-career-score">${alt.score.toFixed(1)}% Match</span>
                    `;
                    altContainer.appendChild(altItem);
                });

                // C. Inject Recommendations Checklist
                const recContainer = document.getElementById("recommendations-list");
                recContainer.innerHTML = "";
                
                const recIcons = {
                    "success": "fa-circle-check",
                    "danger": "fa-triangle-exclamation",
                    "warning": "fa-circle-exclamation",
                    "info": "fa-circle-info"
                };

                pred.recommendations.forEach(rec => {
                    const recLi = document.createElement("li");
                    recLi.className = `rec-item rec-${rec.type}`;
                    recLi.innerHTML = `
                        <i class="fa-solid ${recIcons[rec.type] || 'fa-circle-info'}"></i>
                        <span>${rec.text}</span>
                    `;
                    recContainer.appendChild(recLi);
                });

                // Update Timestamp
                const timeString = new Date().toLocaleTimeString("id-ID", { hour: '2-digit', minute: '2-digit', second: '2-digit' });
                document.getElementById("result-timestamp").textContent = `Prediksi Pukul ${timeString}`;
                
            } else {
                const errData = await res.json();
                alert(`Gagal memproses prediksi: ${errData.error}`);
                resetSinglePlaceholder();
            }
        } catch (err) {
            console.error("Error during prediction fetch:", err);
            alert("Terjadi kesalahan jaringan saat memproses prediksi.");
            resetSinglePlaceholder();
        }
    });

    const resetSinglePlaceholder = () => {
        resultPlaceholder.innerHTML = `
            <div class="placeholder-icon"><i class="fa-solid fa-gears"></i></div>
            <h3>Gagal Memproses</h3>
            <p>Silakan coba jalankan ulang formulir input profil mahasiswa.</p>
        `;
        resultPlaceholder.classList.remove("hidden");
        resultContent.classList.add("hidden");
    };

    // --- TAB 3: BATCH PREDICT CSV UPLOADER ---
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("batch-file-input");
    const dropPrompt = document.querySelector(".drop-zone-prompt");
    const batchResultsPanel = document.getElementById("batch-results-panel");
    const btnDownloadBatch = document.getElementById("btn-download-batch");

    // Click drop-zone to open explorer
    dropZone.addEventListener("click", () => fileInput.click());

    // Sync input change
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length) {
            processCSVFile(e.target.files[0]);
        }
    });

    // Drag-over styling
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add("drop-zone--over");
        }, false);
    });

    ["dragleave", "dragend"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove("drop-zone--over");
        }, false);
    });

    // File drop
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drop-zone--over");
        
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            processCSVFile(e.dataTransfer.files[0]);
        }
    });

    const processCSVFile = async (file) => {
        if (!file.name.endsWith(".csv")) {
            alert("File yang diunggah harus memiliki format .csv");
            return;
        }

        // Show uploading prompt
        dropZone.classList.add("drop-zone--over");
        dropPrompt.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Sedang mengunggah dan menganalisis ${file.name}...`;

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/predict_batch", {
                method: "POST",
                body: formData
            });

            if (res.ok) {
                // Read CSV response as text
                const csvText = await res.text();
                
                // Save it to a global Blob for the download button
                batchBlob = new Blob([csvText], { type: "text/csv;charset=utf-8;" });
                batchFilename = `hasil_prediksi_${file.name}`;
                
                // Calculate metrics client-side from returned prediction CSV
                const lines = csvText.split("\n").map(l => l.trim()).filter(l => l.length > 0);
                const header = lines[0].split(",");
                
                // Find column indexes of predictions
                const idxPredGrad = header.indexOf("Prediksi_Kelulusan");
                
                let totalRows = lines.length - 1; // Subtract header
                let onTimeCount = 0;
                let delayedCount = 0;

                for (let i = 1; i < lines.length; i++) {
                    const row = lines[i].split(",");
                    if (row[idxPredGrad] === "Tepat Waktu") {
                        onTimeCount++;
                    } else if (row[idxPredGrad] === "Tidak Tepat Waktu") {
                        delayedCount++;
                    }
                }

                // Update uploader layout UI
                dropZone.classList.remove("drop-zone--over");
                dropPrompt.innerHTML = `<i class="fa-solid fa-circle-check" style="color:var(--color-success)"></i> Berhasil dianalisis! Tarik file baru untuk memproses ulang.`;
                
                // Update Batch results pane
                document.getElementById("batch-file-name-label").textContent = file.name;
                document.getElementById("batch-total-rows").textContent = totalRows;
                document.getElementById("batch-on-time-rows").textContent = onTimeCount;
                document.getElementById("batch-delayed-rows").textContent = delayedCount;
                
                batchResultsPanel.classList.remove("hidden");
                
            } else {
                const errData = await res.json();
                alert(`Gagal memproses batch: ${errData.error}`);
                resetUploaderPrompt();
            }
        } catch (err) {
            console.error("Error during batch fetch:", err);
            alert("Terjadi kesalahan saat memproses data batch.");
            resetUploaderPrompt();
        }
    };

    const resetUploaderPrompt = () => {
        dropZone.classList.remove("drop-zone--over");
        dropPrompt.textContent = "Tarik dan letakkan file CSV Anda di sini atau klik untuk menjelajah";
        batchResultsPanel.classList.add("hidden");
    };

    // Download batch csv button
    btnDownloadBatch.addEventListener("click", () => {
        if (!batchBlob) return;
        
        const link = document.createElement("a");
        const url = URL.createObjectURL(batchBlob);
        link.setAttribute("href", url);
        link.setAttribute("download", batchFilename);
        link.style.visibility = "hidden";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

});
