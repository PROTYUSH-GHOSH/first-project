/**
 * RoadGuard AI - Active Learning & Interactive Human Verification Studio
 * Fixed: Clean base image + dynamic overlay so deleted predictions vanish immediately!
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const mainDisplay = document.getElementById("mainDisplay");
    const annotationCanvas = document.getElementById("annotationCanvas");
    const emptyPlaceholder = document.getElementById("emptyPlaceholder");
    const loader = document.getElementById("loader");
    const loaderText = document.getElementById("loaderText");
    const samplePills = document.getElementById("samplePills");
    const engineSelect = document.getElementById("engineSelect");
    const conditionBadge = document.getElementById("conditionBadge");
    const conditionText = document.getElementById("conditionText");
    const riskScore = document.getElementById("riskScore");
    const riskBar = document.getElementById("riskBar");
    const totalHazardsCount = document.getElementById("totalHazardsCount");
    const directivesList = document.getElementById("directivesList");

    // Correction Toolbar Elements
    const btnToggleCorrection = document.getElementById("btnToggleCorrection");
    const toggleCorrectionText = document.getElementById("toggleCorrectionText");
    const correctionToolbar = document.getElementById("correctionToolbar");
    const drawClassSelect = document.getElementById("drawClassSelect");
    const btnSaveVerified = document.getElementById("btnSaveVerified");

    // Counts
    const countPothole = document.getElementById("count-pothole");
    const countCrack = document.getElementById("count-crack");
    const countDebris = document.getElementById("count-debris");
    const countWater = document.getElementById("count-water");
    const countBarrier = document.getElementById("count-barrier");

    // Stats
    const trainCount = document.getElementById("trainCount");
    const yoloWeightsStatus = document.getElementById("yoloWeightsStatus");
    const verifiedBadgeCount = document.getElementById("verifiedBadgeCount");

    // Trainer Modal Elements
    const trainerModal = document.getElementById("trainerModal");
    const btnOpenTrainer = document.getElementById("btnOpenTrainer");
    const btnCloseTrainer = document.getElementById("btnCloseTrainer");
    const btnQuickTrainLink = document.getElementById("btnQuickTrainLink");
    const presetBtns = document.querySelectorAll(".preset-btn");
    const btnStartTraining = document.getElementById("btnStartTraining");
    const trainingStatusBadge = document.getElementById("trainingStatusBadge");
    const progressEpochText = document.getElementById("progressEpochText");
    const progressPctText = document.getElementById("progressPctText");
    const hudProgressFill = document.getElementById("hudProgressFill");
    const tLoss = document.getElementById("tLoss");
    const tElapsed = document.getElementById("tElapsed");
    const tVerified = document.getElementById("tVerified");
    const terminalLogs = document.getElementById("terminalLogs");
    const btnSwitchRoboflow = document.getElementById("btnSwitchRoboflow");
    const btnSwitchRDD = document.getElementById("btnSwitchRDD");
    const btnSwitchCustom = document.getElementById("btnSwitchCustom");

    const chkUseMemory = document.getElementById("chkUseMemory");

    // State Variables
    let currentCleanImageB64 = null;
    let originalImageWidth = 800;
    let originalImageHeight = 600;
    let editableBoxes = [];
    let isCorrectionMode = false;
    let isDrawing = false;
    let lastIsVerified = false;
    let startX = 0;
    let startY = 0;
    let currentX = 0;
    let currentY = 0;
    let availableSamples = [];
    let trainingPollInterval = null;
    let selectedEpochs = 3;
    let selectedBatch = 4;

    const CLASS_COLORS = {
        "pothole": "#ef4444",
        "road_crack": "#f97316",
        "debris": "#eab308",
        "standing_water": "#06b6d4",
        "construction_barrier": "#fb923c"
    };

    // --- Modal Controls ---
    function openTrainer() {
        trainerModal.style.display = "flex";
        loadStats();
    }
    function closeTrainer() {
        trainerModal.style.display = "none";
    }

    btnOpenTrainer.addEventListener("click", openTrainer);
    btnQuickTrainLink.addEventListener("click", openTrainer);
    btnCloseTrainer.addEventListener("click", closeTrainer);
    trainerModal.addEventListener("click", (e) => {
        if (e.target === trainerModal) closeTrainer();
    });

    // --- Training Presets ---
    presetBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            presetBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            selectedEpochs = parseInt(btn.getAttribute("data-epochs"));
            selectedBatch = parseInt(btn.getAttribute("data-batch"));
        });
    });

    // --- Model Switching (Roboflow vs RDD vs Custom) ---
    async function switchModel(type) {
        try {
            if (type === "roboflow") {
                engineSelect.value = "roboflow";
                btnSwitchRoboflow.classList.add("active");
                btnSwitchRDD.classList.remove("active");
                btnSwitchCustom.classList.remove("active");
                yoloWeightsStatus.textContent = "Roboflow Cloud (pothole-detection-jnjlz)";
            } else {
                const res = await fetch("/api/model/switch", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ model_type: type })
                });
                const data = await res.json();
                if (type === "rdd") {
                    btnSwitchRDD.classList.add("active");
                    btnSwitchCustom.classList.remove("active");
                    btnSwitchRoboflow.classList.remove("active");
                    engineSelect.value = "yolo_only";
                } else {
                    btnSwitchCustom.classList.add("active");
                    btnSwitchRDD.classList.remove("active");
                    btnSwitchRoboflow.classList.remove("active");
                    engineSelect.value = "yolo_only";
                }
                loadStats();
            }

            if (availableSamples.length > 0) {
                runSampleDetection(availableSamples[0], samplePills.children[0]);
            }
        } catch (e) {
            console.error("Model switch failed", e);
        }
    }

    if (btnSwitchRoboflow) btnSwitchRoboflow.addEventListener("click", () => switchModel("roboflow"));
    btnSwitchRDD.addEventListener("click", () => switchModel("rdd"));
    btnSwitchCustom.addEventListener("click", () => switchModel("custom"));

    // --- Start Training ---
    btnStartTraining.addEventListener("click", async () => {
        btnStartTraining.disabled = true;
        btnStartTraining.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Initializing Neural Pipeline...`;
        trainingStatusBadge.textContent = "TRAINING";
        trainingStatusBadge.className = "badge-status training";

        try {
            const res = await fetch("/api/train/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ epochs: selectedEpochs, batch_size: selectedBatch, imgsz: 512 })
            });
            const data = await res.json();
            appendLog(data.message);
            startTrainingPolling();
        } catch (e) {
            alert("Failed to start training: " + e.message);
            btnStartTraining.disabled = false;
            btnStartTraining.innerHTML = `<i class="fa-solid fa-play"></i> Train AI on Verified Dataset`;
        }
    });

    // --- Training Telemetry Polling ---
    function startTrainingPolling() {
        if (trainingPollInterval) clearInterval(trainingPollInterval);
        
        trainingPollInterval = setInterval(async () => {
            try {
                const res = await fetch("/api/train/status");
                const state = await res.json();

                progressEpochText.textContent = `Epoch ${state.epoch} / ${state.total_epochs}`;
                progressPctText.textContent = `${state.progress_pct}%`;
                hudProgressFill.style.width = `${state.progress_pct}%`;
                tLoss.textContent = state.loss || "--";
                tElapsed.textContent = `${state.elapsed_seconds || 0}s`;

                if (state.logs && state.logs.length > 0) {
                    terminalLogs.innerHTML = "";
                    state.logs.forEach(line => {
                        const p = document.createElement("p");
                        p.className = "log-line";
                        p.textContent = line;
                        terminalLogs.appendChild(p);
                    });
                    terminalLogs.scrollTop = terminalLogs.scrollHeight;
                }

                if (!state.is_training && state.progress_pct >= 100) {
                    clearInterval(trainingPollInterval);
                    trainingPollInterval = null;
                    trainingStatusBadge.textContent = "COMPLETED";
                    trainingStatusBadge.className = "badge-status done";
                    btnStartTraining.disabled = false;
                    btnStartTraining.innerHTML = `<i class="fa-solid fa-check"></i> Training Complete! Retrain?`;
                    
                    btnSwitchCustom.classList.add("active");
                    btnSwitchRDD.classList.remove("active");
                    loadStats();
                    appendLog("🎉 New model fine-tuned on verified data & hot-reloaded into vision feed!");
                }
            } catch (e) {
                console.error("Poll error", e);
            }
        }, 1500);
    }

    function appendLog(msg) {
        const p = document.createElement("p");
        p.className = "log-line";
        p.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        terminalLogs.appendChild(p);
        terminalLogs.scrollTop = terminalLogs.scrollHeight;
    }

    // --- Fetch Dataset & System Stats ---
    async function loadStats() {
        try {
            const res = await fetch("/api/stats");
            const data = await res.json();
            trainCount.textContent = `${data.verified_human_samples || 0} photos verified`;
            yoloWeightsStatus.textContent = data.active_model_name;
            verifiedBadgeCount.textContent = `${data.verified_human_samples || 0} Human-Verified Real Photos`;
            tVerified.textContent = data.verified_human_samples || 0;

            if (data.active_model_type === "rdd_real_world") {
                btnSwitchRDD.classList.add("active");
                btnSwitchCustom.classList.remove("active");
            } else if (data.active_model_type === "fine_tuned") {
                btnSwitchCustom.classList.add("active");
                btnSwitchRDD.classList.remove("active");
            }

            if (data.is_training && !trainingPollInterval) {
                btnStartTraining.disabled = true;
                btnStartTraining.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Training In Progress...`;
                trainingStatusBadge.textContent = "TRAINING";
                trainingStatusBadge.className = "badge-status training";
                startTrainingPolling();
            }
        } catch (e) {
            console.error("Failed to fetch stats", e);
        }
    }

    // --- Fetch Showcase Samples ---
    async function loadSamples() {
        try {
            const res = await fetch("/api/samples");
            availableSamples = await res.json();
            samplePills.innerHTML = "";
            availableSamples.forEach((sample, idx) => {
                const pill = document.createElement("button");
                pill.className = "sample-pill";
                pill.textContent = `Scene ${idx + 1}`;
                pill.onclick = () => runSampleDetection(sample, pill);
                samplePills.appendChild(pill);
            });
            if (availableSamples.length > 0) {
                runSampleDetection(availableSamples[0], samplePills.children[0]);
            }
        } catch (e) {
            console.error("Failed to load samples", e);
        }
    }

    // --- Run Detection on Sample ---
    async function runSampleDetection(filename, pillElement) {
        showLoader(`Analyzing scene (${filename})...`);
        document.querySelectorAll(".sample-pill").forEach(p => p.classList.remove("active"));
        if (pillElement) pillElement.classList.add("active");

        try {
            const engine = engineSelect.value;
            const bypass = chkUseMemory && !chkUseMemory.checked ? "&bypass_memory=true" : "";
            const res = await fetch(`/api/detect?engine=${engine}${bypass}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sample_filename: filename })
            });
            const data = await res.json();
            updateUI(data);
        } catch (e) {
            alert("Error analyzing image: " + e.message);
        } finally {
            hideLoader();
        }
    }

    // --- Run Detection on File Upload ---
    async function uploadAndDetect(file) {
        showLoader(`Evaluating: ${file.name}...`);
        const engine = engineSelect.value;
        const bypass = chkUseMemory && !chkUseMemory.checked ? "&bypass_memory=true" : "";
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch(`/api/detect?engine=${engine}${bypass}`, {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            updateUI(data);
            toggleCorrectionMode(true);
        } catch (e) {
            alert("Upload failed: " + e.message);
        } finally {
            hideLoader();
        }
    }

    // --- Update UI with Detection Results ---
    function updateUI(data) {
        if (!data || !data.result) return;
        const res = data.result;

        currentCleanImageB64 = data.clean_image_b64 || data.original_image_b64;
        originalImageWidth = data.image_dimensions?.width || 800;
        originalImageHeight = data.image_dimensions?.height || 600;
        lastIsVerified = !!data.is_verified_by_user;

        // Store raw detections as editable boxes
        editableBoxes = (data.raw_detections || []).map((d, index) => ({
            id: index + 1,
            class: d.class,
            bbox: [...d.bbox],
            confidence: d.confidence
        }));

        // Set clean image to mainDisplay
        mainDisplay.src = currentCleanImageB64;
        mainDisplay.style.display = "block";
        emptyPlaceholder.style.display = "none";
        annotationCanvas.style.display = "block";

        mainDisplay.onload = () => {
            alignAndRenderCanvas();
        };

        syncTelemetryFromBoxes(res.condition, res.condition_color);

        directivesList.innerHTML = "";
        if (data.recommendations && data.recommendations.length > 0) {
            data.recommendations.forEach(rec => {
                const card = document.createElement("div");
                card.className = `directive-card priority-${rec.priority}`;
                card.innerHTML = `<strong>[${rec.priority}]</strong> ${rec.action}`;
                directivesList.appendChild(card);
            });
        }
    }

    // Recalculates hazard counters on the right sidebar when user deletes/adds boxes
    function syncTelemetryFromBoxes(forceCond = null, forceColor = null) {
        const counts = {
            "pothole": 0,
            "road_crack": 0,
            "debris": 0,
            "standing_water": 0,
            "construction_barrier": 0
        };

        editableBoxes.forEach(b => {
            counts[b.class] = (counts[b.class] || 0) + 1;
        });

        countPothole.textContent = counts["pothole"];
        countCrack.textContent = counts["road_crack"];
        countDebris.textContent = counts["debris"];
        countWater.textContent = counts["standing_water"];
        countBarrier.textContent = counts["construction_barrier"];

        const total = editableBoxes.length;
        totalHazardsCount.textContent = `${total} Detected`;

        // Calculate score
        let score = (counts["pothole"] * 30) + (counts["debris"] * 25) + (counts["road_crack"] * 15) + (counts["standing_water"] * 15) + (counts["construction_barrier"] * 10);
        score = Math.min(100, score);
        riskScore.textContent = `${score} / 100`;
        riskBar.style.width = `${score}%`;

        if (lastIsVerified) {
            conditionText.innerHTML = `<i class="fa-solid fa-circle-check"></i> VERIFIED BY YOU`;
            conditionBadge.style.borderColor = "#10b981";
            conditionBadge.style.backgroundColor = "rgba(16, 185, 129, 0.25)";
            conditionText.style.color = "#34d399";
        } else {
            const finalCond = forceCond || (score > 60 ? "DANGEROUS" : score > 30 ? "CAUTION ADVISED" : score > 0 ? "FAIR / MONITOR" : "CLEAR & SAFE");
            const finalColor = forceColor || (score > 60 ? "#ef4444" : score > 30 ? "#f59e0b" : score > 0 ? "#3b82f6" : "#10b981");
            conditionText.textContent = finalCond;
            conditionBadge.style.borderColor = finalColor;
            conditionBadge.style.backgroundColor = "rgba(11, 15, 25, 0.85)";
            conditionText.style.color = finalColor;
        }
    }

    // ========================================================
    // --- CANVAS SIZING, ALIGNMENT & RENDERING ---
    // ========================================================

    function alignAndRenderCanvas() {
        if (!mainDisplay.complete || mainDisplay.naturalWidth === 0) return;

        const w = mainDisplay.clientWidth;
        const h = mainDisplay.clientHeight;
        const left = mainDisplay.offsetLeft;
        const top = mainDisplay.offsetTop;

        annotationCanvas.width = w;
        annotationCanvas.height = h;
        annotationCanvas.style.width = `${w}px`;
        annotationCanvas.style.height = `${h}px`;
        annotationCanvas.style.left = `${left}px`;
        annotationCanvas.style.top = `${top}px`;

        renderCanvas();
    }

    function renderCanvas() {
        const ctx = annotationCanvas.getContext("2d");
        ctx.clearRect(0, 0, annotationCanvas.width, annotationCanvas.height);

        const rectW = annotationCanvas.width;
        const rectH = annotationCanvas.height;

        const scaleX = rectW / floatSafe(originalImageWidth);
        const scaleY = rectH / floatSafe(originalImageHeight);

        // Draw bounding boxes on canvas
        editableBoxes.forEach(b => {
            const x1 = Math.round(b.bbox[0] * scaleX);
            const y1 = Math.round(b.bbox[1] * scaleY);
            const x2 = Math.round(b.bbox[2] * scaleX);
            const y2 = Math.round(b.bbox[3] * scaleY);
            const bw = x2 - x1;
            const bh = y2 - y1;
            const color = CLASS_COLORS[b.class] || "#10b981";

            // Box outline
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;
            ctx.strokeRect(x1, y1, bw, bh);

            // Shaded fill
            ctx.fillStyle = hexToRgba(color, 0.18);
            ctx.fillRect(x1, y1, bw, bh);

            // Label tag badge
            const label = b.class.replace("_", " ").toUpperCase();
            const confTag = b.confidence ? ` ${Math.round(b.confidence * 100)}%` : "";
            const fullTag = `${label}${confTag}`;

            ctx.font = "bold 11px sans-serif";
            const textWidth = ctx.measureText(fullTag).width;
            const badgeH = 20;
            const badgeY = Math.max(0, y1 - badgeH);

            ctx.fillStyle = color;
            ctx.fillRect(x1, badgeY, textWidth + 12, badgeH);
            ctx.fillStyle = "#ffffff";
            ctx.fillText(fullTag, x1 + 6, badgeY + 14);

            // In Review Mode: Draw red 'X' delete button on top-right of box
            if (isCorrectionMode) {
                const btnSize = 22;
                const btnX = x2 - btnSize;
                const btnY = Math.max(0, y1 - btnSize);

                // Circular shadow & red pill
                ctx.fillStyle = "#ef4444";
                ctx.beginPath();
                ctx.roundRect(btnX, btnY, btnSize, btnSize, 4);
                ctx.fill();

                ctx.fillStyle = "#ffffff";
                ctx.font = "bold 13px sans-serif";
                ctx.fillText("✕", btnX + 6, btnY + 16);

                // Store exact hit area for click listener
                b._hitDel = { x1: btnX, y1: btnY, x2: btnX + btnSize, y2: btnY + btnSize };
            } else {
                b._hitDel = null;
            }
        });

        // Draw actively dragged box in review mode
        if (isDrawing && isCorrectionMode) {
            const curW = currentX - startX;
            const curH = currentY - startY;
            const chosenClass = drawClassSelect.value;
            const activeColor = CLASS_COLORS[chosenClass] || "#3b82f6";

            ctx.strokeStyle = activeColor;
            ctx.setLineDash([5, 4]);
            ctx.lineWidth = 2.5;
            ctx.strokeRect(startX, startY, curW, curH);
            ctx.setLineDash([]);

            ctx.fillStyle = hexToRgba(activeColor, 0.25);
            ctx.fillRect(startX, startY, curW, curH);
        }
    }

    function floatSafe(val) {
        return val > 0 ? val : 1;
    }

    function hexToRgba(hex, alpha) {
        let c = hex.replace("#", "");
        if (c.length === 3) c = c.split("").map(x => x + x).join("");
        const num = parseInt(c, 16);
        return `rgba(${(num >> 16) & 255}, ${(num >> 8) & 255}, ${num & 255}, ${alpha})`;
    }

    // Toggle Review Mode
    function toggleCorrectionMode(forceOn = null) {
        isCorrectionMode = forceOn !== null ? forceOn : !isCorrectionMode;

        if (isCorrectionMode) {
            btnToggleCorrection.classList.add("active");
            toggleCorrectionText.textContent = "Done Reviewing";
            correctionToolbar.style.display = "flex";
            annotationCanvas.style.cursor = "crosshair";
        } else {
            btnToggleCorrection.classList.remove("active");
            toggleCorrectionText.textContent = "Review & Correct AI";
            correctionToolbar.style.display = "none";
            annotationCanvas.style.cursor = "default";
        }
        alignAndRenderCanvas();
    }

    btnToggleCorrection.addEventListener("click", () => toggleCorrectionMode());

    // Canvas Click Listener for Deleting & Drawing
    annotationCanvas.addEventListener("mousedown", (e) => {
        if (!isCorrectionMode) return;

        const rect = annotationCanvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        // Check if user clicked any delete button '✕'
        for (let i = editableBoxes.length - 1; i >= 0; i--) {
            const hit = editableBoxes[i]._hitDel;
            if (hit && mouseX >= hit.x1 && mouseX <= hit.x2 && mouseY >= hit.y1 && mouseY <= hit.y2) {
                // Delete box immediately!
                editableBoxes.splice(i, 1);
                syncTelemetryFromBoxes();
                renderCanvas();
                return; // Stop event so we don't start drawing
            }
        }

        // Start drawing new box
        isDrawing = true;
        startX = mouseX;
        startY = mouseY;
        currentX = mouseX;
        currentY = mouseY;
    });

    annotationCanvas.addEventListener("mousemove", (e) => {
        if (!isDrawing || !isCorrectionMode) return;
        const rect = annotationCanvas.getBoundingClientRect();
        currentX = e.clientX - rect.left;
        currentY = e.clientY - rect.top;
        renderCanvas();
    });

    annotationCanvas.addEventListener("mouseup", () => {
        if (!isDrawing || !isCorrectionMode) return;
        isDrawing = false;

        const rect = annotationCanvas.getBoundingClientRect();
        const scaleX = originalImageWidth / floatSafe(rect.width);
        const scaleY = originalImageHeight / floatSafe(rect.height);

        const x1 = Math.min(startX, currentX) * scaleX;
        const y1 = Math.min(startY, currentY) * scaleY;
        const x2 = Math.max(startX, currentX) * scaleX;
        const y2 = Math.max(startY, currentY) * scaleY;

        // Add box if at least 15px
        if ((x2 - x1) > 15 && (y2 - y1) > 15) {
            const selectedClass = drawClassSelect.value;
            editableBoxes.push({
                id: Date.now(),
                class: selectedClass,
                bbox: [Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2)],
                confidence: 1.0 // Verified
            });
            syncTelemetryFromBoxes();
        }

        renderCanvas();
    });

    // Save Verified Sample to Backend Training Set
    btnSaveVerified.addEventListener("click", async () => {
        if (!currentCleanImageB64) {
            alert("No image is currently loaded.");
            return;
        }

        btnSaveVerified.disabled = true;
        btnSaveVerified.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Saving Verified Corrections...`;

        try {
            const payload = {
                image_base64: currentCleanImageB64,
                boxes: editableBoxes
            };

            const res = await fetch("/api/dataset/submit-verified", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            btnSaveVerified.disabled = false;
            btnSaveVerified.innerHTML = `<i class="fa-solid fa-check-double"></i> Verified & Added!`;
            btnSaveVerified.style.backgroundColor = "#059669";

            appendLog(`✅ Verified Sample Saved! (${editableBoxes.length} confirmed hazards)`);
            loadStats();

            setTimeout(() => {
                btnSaveVerified.innerHTML = `<i class="fa-solid fa-check"></i> Save & Add to Training Set`;
                btnSaveVerified.style.backgroundColor = "";
            }, 3000);

        } catch (e) {
            alert("Save failed: " + e.message);
            btnSaveVerified.disabled = false;
            btnSaveVerified.innerHTML = `<i class="fa-solid fa-check"></i> Save & Add to Training Set`;
        }
    });

    window.addEventListener("resize", () => {
        alignAndRenderCanvas();
    });

    function showLoader(msg) {
        loaderText.textContent = msg || "Running neural detection filters...";
        loader.style.display = "flex";
    }

    function hideLoader() {
        loader.style.display = "none";
    }

    // --- Drag and Drop File Upload ---
    dropZone.addEventListener("click", (e) => {
        if (isCorrectionMode) return;
        fileInput.click();
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            uploadAndDetect(e.target.files[0]);
        }
    });

    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            uploadAndDetect(files[0]);
        }
    });

    // --- Engine Switch Event ---
    engineSelect.addEventListener("change", () => {
        if (availableSamples.length > 0) {
            runSampleDetection(availableSamples[0], samplePills.children[0]);
        }
    });

    // --- Initialization ---
    loadStats();
    loadSamples();
});
