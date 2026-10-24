/**
 * AI Email Classifier - Client-side Interactive Scripts
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------
    // 1. Elements
    // -------------------------------------------------------------
    const form = document.getElementById("emailClassifierForm");
    const emailInput = document.getElementById("emailInput");
    const clearBtn = document.getElementById("clearBtn");
    const classifyBtn = document.getElementById("classifyBtn");
    const classifySpinner = document.getElementById("classifySpinner");
    const classifyIcon = document.getElementById("classifyIcon");
    const inputValidationMsg = document.getElementById("inputValidationMsg");

    const resultPlaceholder = document.getElementById("resultPlaceholder");
    const resultContent = document.getElementById("resultContent");
    const predictionBanner = document.getElementById("predictionBanner");
    const predictedCategoryText = document.getElementById("predictedCategoryText");
    const predictionIcon = document.getElementById("predictionIcon");
    const predictionConfidence = document.getElementById("predictionConfidence");
    const confidenceBarLabel = document.getElementById("confidenceBarLabel");
    const confidenceProgressBar = document.getElementById("confidenceProgressBar");
    const probabilityBreakdownList = document.getElementById("probabilityBreakdownList");
    const explanationText = document.getElementById("explanationText");

    const clearHistoryBtn = document.getElementById("clearHistoryBtn");
    const historyTableBody = document.getElementById("historyTableBody");
    const noHistoryRow = document.getElementById("noHistoryRow");

    const datasetSearchInput = document.getElementById("datasetSearchInput");
    const datasetSampleTable = document.getElementById("datasetSampleTable");

    // Category styling configurations
    const categoryConfigs = {
        Work: {
            bgClass: "bg-primary-subtle text-primary border border-primary",
            badgeClass: "bg-primary-subtle text-primary border-primary",
            progressClass: "bg-primary",
            icon: "bi-briefcase-fill"
        },
        Personal: {
            bgClass: "bg-info-subtle text-info-emphasis border border-info",
            badgeClass: "bg-info-subtle text-info-emphasis border-info",
            progressClass: "bg-info",
            icon: "bi-people-fill"
        },
        Promotions: {
            bgClass: "bg-warning-subtle text-warning-emphasis border border-warning",
            badgeClass: "bg-warning-subtle text-warning-emphasis border-warning",
            progressClass: "bg-warning",
            icon: "bi-tag-fill"
        },
        Spam: {
            bgClass: "bg-danger-subtle text-danger border border-danger",
            badgeClass: "bg-danger-subtle text-danger border-danger",
            progressClass: "bg-danger",
            icon: "bi-shield-slash-fill"
        },
        Important: {
            bgClass: "bg-success-subtle text-success border border-success",
            badgeClass: "bg-success-subtle text-success border-success",
            progressClass: "bg-success",
            icon: "bi-exclamation-triangle-fill"
        }
    };

    // -------------------------------------------------------------
    // 2. Quick Test Sample Buttons
    // -------------------------------------------------------------
    const sampleButtons = document.querySelectorAll(".sample-btn");
    sampleButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
            const sampleText = btn.getAttribute("data-text");
            if (emailInput && sampleText) {
                emailInput.value = sampleText;
                emailInput.classList.remove("is-invalid");
                emailInput.focus();
                // Automatically submit form for instant demonstration
                submitClassification();
            }
        });
    });

    // -------------------------------------------------------------
    // 3. Clear Button Handler
    // -------------------------------------------------------------
    if (clearBtn && emailInput) {
        clearBtn.addEventListener("click", () => {
            emailInput.value = "";
            emailInput.classList.remove("is-invalid");
            emailInput.focus();
            if (resultPlaceholder && resultContent) {
                resultPlaceholder.classList.remove("d-none");
                resultContent.classList.add("d-none");
            }
        });
    }

    // -------------------------------------------------------------
    // 4. Form Submission and Classification
    // -------------------------------------------------------------
    if (form) {
        form.addEventListener("submit", (e) => {
            e.preventDefault();
            submitClassification();
        });
    }

    async function submitClassification() {
        if (!emailInput) return;
        const rawText = emailInput.value.trim();

        // Validate non-empty alphanumeric input
        if (!rawText || !/[a-zA-Z0-9]/.test(rawText)) {
            emailInput.classList.add("is-invalid");
            if (inputValidationMsg) {
                inputValidationMsg.textContent = "Please enter meaningful email text (cannot be empty or symbols only).";
            }
            emailInput.focus();
            return;
        }

        emailInput.classList.remove("is-invalid");

        // Set Loading state
        setLoadingState(true);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Requested-With": "XMLHttpRequest"
                },
                body: JSON.stringify({ email_text: rawText })
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || "An error occurred during classification.");
            }

            renderPrediction(data);
            prependToHistory(data);

        } catch (err) {
            alert(err.message || "Failed to classify email. Please try again.");
        } finally {
            setLoadingState(false);
        }
    }

    function setLoadingState(isLoading) {
        if (!classifyBtn) return;
        classifyBtn.disabled = isLoading;
        if (classifySpinner) {
            classifySpinner.classList.toggle("d-none", !isLoading);
        }
        if (classifyIcon) {
            classifyIcon.classList.toggle("d-none", isLoading);
        }
    }

    function renderPrediction(data) {
        if (!resultPlaceholder || !resultContent) return;

        resultPlaceholder.classList.add("d-none");
        resultContent.classList.remove("d-none");

        const category = data.predicted_category;
        const confidence = data.confidence;
        const config = categoryConfigs[category] || {
            bgClass: "bg-secondary text-white",
            progressClass: "bg-secondary",
            icon: "bi-envelope"
        };

        // Update Banner
        predictionBanner.className = `text-center mb-4 p-4 rounded-4 ${config.bgClass}`;
        predictedCategoryText.textContent = category;
        predictionIcon.className = `bi ${config.icon} fs-2`;
        predictionConfidence.textContent = `${confidence}%`;

        // Update Top Confidence Progress Bar
        confidenceBarLabel.textContent = `${confidence}%`;
        confidenceProgressBar.className = `progress-bar progress-bar-striped progress-bar-animated ${config.progressClass}`;
        confidenceProgressBar.style.width = `${confidence}%`;

        // Update Probability Breakdown
        if (probabilityBreakdownList && data.all_probabilities) {
            probabilityBreakdownList.innerHTML = "";
            for (const [catName, prob] of Object.entries(data.all_probabilities)) {
                const itemConfig = categoryConfigs[catName] || { progressClass: "bg-secondary" };
                const isWinner = catName === category;

                const item = document.createElement("div");
                item.className = `list-group-item d-flex align-items-center justify-content-between py-2 px-3 border-0 rounded-2 mb-1 ${
                    isWinner ? "bg-white shadow-xs fw-semibold" : "bg-transparent text-muted"
                }`;

                item.innerHTML = `
                    <div class="d-flex align-items-center gap-2" style="width: 35%;">
                        <i class="bi ${itemConfig.icon}"></i>
                        <span>${catName}</span>
                    </div>
                    <div class="progress flex-grow-1 mx-3" style="height: 6px;">
                        <div class="progress-bar ${itemConfig.progressClass}" style="width: ${prob}%"></div>
                    </div>
                    <div class="text-end font-monospace small" style="width: 15%;">
                        ${prob}%
                    </div>
                `;
                probabilityBreakdownList.appendChild(item);
            }
        }

        // Update Explanation
        if (explanationText && data.explanation) {
            explanationText.textContent = data.explanation;
        }
    }

    function prependToHistory(data) {
        if (!historyTableBody) return;

        // Hide "No history" row if present
        if (noHistoryRow) {
            noHistoryRow.classList.add("d-none");
        }

        const config = categoryConfigs[data.predicted_category] || {
            badgeClass: "bg-secondary text-white",
            icon: "bi-envelope",
            progressClass: "bg-primary"
        };

        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

        const newRow = document.createElement("tr");
        newRow.innerHTML = `
            <td class="small text-muted font-monospace">${timeStr}</td>
            <td class="text-truncate" style="max-width: 320px;" title="${escapeHtml(data.input_preview)}">
                ${escapeHtml(data.input_preview)}
            </td>
            <td>
                <span class="badge ${config.badgeClass} rounded-pill px-3 py-1">
                    <i class="bi ${config.icon} me-1"></i>${data.predicted_category}
                </span>
            </td>
            <td>
                <div class="d-flex align-items-center gap-2">
                    <span class="small fw-semibold">${data.confidence}%</span>
                    <div class="progress flex-grow-1" style="height: 6px; width: 60px;">
                        <div class="progress-bar ${config.progressClass}" style="width: ${data.confidence}%"></div>
                    </div>
                </div>
            </td>
        `;

        historyTableBody.insertBefore(newRow, historyTableBody.firstChild);

        // Enable Clear History Button
        if (clearHistoryBtn) {
            clearHistoryBtn.disabled = false;
        }
    }

    // -------------------------------------------------------------
    // 5. Clear History Action
    // -------------------------------------------------------------
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener("click", async () => {
            try {
                const response = await fetch("/clear-history", { method: "POST" });
                if (response.ok) {
                    if (historyTableBody) {
                        historyTableBody.innerHTML = `
                            <tr id="noHistoryRow">
                                <td colspan="4" class="text-center text-muted py-4 small">
                                    No emails classified yet in this session. Classify an email above to view history!
                                </td>
                            </tr>
                        `;
                    }
                    clearHistoryBtn.disabled = true;
                }
            } catch (err) {
                console.error("Failed to clear history", err);
            }
        });
    }

    // -------------------------------------------------------------
    // 6. Dataset Table Search Filter
    // -------------------------------------------------------------
    if (datasetSearchInput && datasetSampleTable) {
        datasetSearchInput.addEventListener("input", () => {
            const query = datasetSearchInput.value.toLowerCase().trim();
            const rows = datasetSampleTable.querySelectorAll(".dataset-row");

            rows.forEach((row) => {
                const text = row.querySelector(".email-text-cell").textContent.toLowerCase();
                const badge = row.querySelector(".badge").textContent.toLowerCase();
                if (text.includes(query) || badge.includes(query)) {
                    row.style.display = "";
                } else {
                    row.style.display = "none";
                }
            });
        });
    }

    // Utility: Simple HTML escaping
    function escapeHtml(str) {
        if (!str) return "";
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
