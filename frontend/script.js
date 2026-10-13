
"use strict";

const API = "http://127.0.0.1:8001";

const state = {
    options: { sector: [], agency: [], state: [], agenciesBySector: {} },
    projects: [],
    sectorData: [],
    yearData: [],
    sectorChart: null,
    yearChart: null,
    toastTimer: null,
    delayDistributionChart: null,
    costHistogramChart: null,
    costDurationChart: null,
};

document.addEventListener("DOMContentLoaded", init);

async function init() {
    setupSidebar();
    setupNavigation();
    setupAssessment();
    setupRefreshButtons();

    const yearInput = document.getElementById("yearInput");
    if (yearInput) yearInput.value = new Date().getFullYear();

    await Promise.allSettled([
        loadOptions(),
        loadSummary(),
        loadAnalytics()
    ]);
}

async function apiGet(path) {
    const response = await fetch(`${API}${path}`, {
        headers: { Accept: "application/json" }
    });
    if (!response.ok) {
        throw new Error(`Request failed (${response.status}) for ${path}`);
    }
    return response.json();
}

async function apiPost(path, data) {
    const response = await fetch(`${API}${path}`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            Accept: "application/json"
        },
        body: JSON.stringify(data)
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(result.detail || `Request failed (${response.status})`);
    }
    return result;
}

/* Sidebar hide/show */
function setupSidebar() {
    const sidebar = document.getElementById("sidebar");
    const hideButton = document.getElementById("sidebarCollapse");
    const openButton = document.getElementById("sidebarOpen");
    if (!sidebar || !hideButton || !openButton) return;

    function hideSidebar() {
        if (window.innerWidth <= 760) {
            sidebar.classList.remove("mobile-open");
            openButton.style.display = "block";
        } else {
            sidebar.classList.add("collapsed");
            openButton.style.display = "none";
        }
    }

    function showSidebar() {
        if (window.innerWidth <= 760) {
            sidebar.classList.add("mobile-open");
            openButton.style.display = "none";
        } else {
            sidebar.classList.remove("collapsed");
            openButton.style.display = "none";
        }
    }

    hideButton.addEventListener("click", hideSidebar);
    openButton.addEventListener("click", showSidebar);

    document.addEventListener("keydown", event => {
        if (event.key === "Escape" && window.innerWidth <= 760) {
            sidebar.classList.remove("mobile-open");
            openButton.style.display = "block";
        }
    });

    window.addEventListener("resize", () => {
        if (window.innerWidth > 760) {
            sidebar.classList.remove("mobile-open");
            openButton.style.display = "none";
        } else if (!sidebar.classList.contains("mobile-open")) {
            openButton.style.display = "block";
        }
    });
}

/* Section navigation */
function setupNavigation() {
    document.querySelectorAll(".nav-item").forEach(button => {
        button.addEventListener("click", () => navigateTo(button.dataset.target));
    });
}

function navigateTo(targetId) {
    const target = document.getElementById(targetId);
    if (!target) return;

    document.querySelectorAll(".nav-item").forEach(button => {
        button.classList.toggle("active", button.dataset.target === targetId);
    });

    target.scrollIntoView({ behavior: "smooth", block: "start" });

    if (window.innerWidth <= 760) {
        const sidebar = document.getElementById("sidebar");
        const openButton = document.getElementById("sidebarOpen");
        sidebar.classList.remove("mobile-open");
        openButton.style.display = "block";
    }
}

/* API options and dependent agency dropdown */
async function loadOptions() {
    try {
        const data = await apiGet("/options");

        state.options = {
            sector: Array.isArray(data.sector) ? data.sector : [],
            agency: Array.isArray(data.agency) ? data.agency : [],
            state: Array.isArray(data.state) ? data.state : [],
            agenciesBySector: data.agencies_by_sector || data.agenciesBySector || {}
        };

        populateSelect("sectorInput", state.options.sector, "Select sector");
        populateSelect("stateInput", state.options.state, "Select state");

        const sectorSelect = document.getElementById("sectorInput");
        sectorSelect.addEventListener("change", updateAgencyOptions);
        updateAgencyOptions();

        setApiStatus(true);
    } catch (error) {
        setApiStatus(false);
        showToast("Could not load dropdown options. Check that FastAPI is running.", true);
        console.error("Loading options failed:", error);
    }
}

function updateAgencyOptions() {
    const sector = document.getElementById("sectorInput").value;
    const agencySelect = document.getElementById("agencyInput");
    const mapping = state.options.agenciesBySector || {};

    let agencies = mapping[sector];

    if (!Array.isArray(agencies)) {
        agencies = state.options.agency;
    }

    populateSelect(
        "agencyInput",
        agencies,
        sector ? "Select agency" : "Select sector first"
    );
}

function populateSelect(id, values, placeholder) {
    const select = document.getElementById(id);
    if (!select) return;

    const oldValue = select.value;
    select.replaceChildren(new Option(placeholder, ""));

    values.forEach(value => {
        select.add(new Option(String(value), String(value)));
    });

    if (values.includes(oldValue)) select.value = oldValue;
}

/* Prediction form */
function setupAssessment() {
    const form = document.getElementById("assessmentForm");
    if (!form) return;

    form.addEventListener("submit", async event => {
        event.preventDefault();

        const button = document.getElementById("predictButton");
        const message = document.getElementById("assessmentMessage");

        button.disabled = true;
        button.textContent = "Predicting…";
        message.textContent = "";

        const payload = {
            sector: document.getElementById("sectorInput").value,
            agency: document.getElementById("agencyInput").value,
            state: document.getElementById("stateInput").value,
            orig_cost_cr: Number(document.getElementById("costInput").value),
            approval_year: Number(document.getElementById("yearInput").value),
            approval_month: Number(document.getElementById("monthInput").value),
            planned_duration_months: Number(document.getElementById("durationInput").value)
        };

        try {
            const result = await apiPost("/predict", payload);
            displayPrediction(result);
            message.textContent = "Prediction completed successfully.";
            message.className = "form-message success";
        } catch (error) {
            message.textContent = error.message || "Unable to generate prediction.";
            message.className = "form-message error";
            showToast(message.textContent, true);
            console.error("Prediction failed:", error);
        } finally {
            button.disabled = false;
            button.textContent = "Predict delay risk →";
        }
    });

    const clearButton = document.getElementById("clearAssessmentButton");
    clearButton.addEventListener("click", () => {
        form.reset();
        document.getElementById("yearInput").value = new Date().getFullYear();

        const empty = document.getElementById("predictionEmpty");
        const result = document.getElementById("predictionResult");
        if (empty) empty.hidden = false;
        if (result) result.hidden = true;

        document.getElementById("assessmentMessage").textContent = "";
    });
}

function displayPrediction(data) {
    const empty = document.getElementById("predictionEmpty");
    const result = document.getElementById("predictionResult");
    if (empty) empty.hidden = true;
    if (result) result.hidden = false;

    let probability = Number(data.delay_probability ?? data.risk_score ?? 0);
    if (probability <= 1) probability *= 100;
    probability = Math.max(0, Math.min(100, probability));

    const scoreValue = Number(data.risk_score ?? probability);
    const score = scoreValue <= 1 ? scoreValue * 100 : scoreValue;
    const level = String(data.risk_level || inferRiskLevel(score)).toUpperCase();

    const badge = document.getElementById("resultRiskLevel");
    badge.textContent = level;
    badge.className = `risk-badge ${riskClass(level)}`;

    document.getElementById("resultProbability").textContent = `${probability.toFixed(1)}%`;
    document.getElementById("resultScore").textContent = `${score.toFixed(1)} / 100`;
    document.getElementById("resultPrediction").textContent =
        data.prediction ||
        (Number(data.predicted_class) === 1 ? "Delay predicted" : "On schedule predicted");

    document.getElementById("resultDescription").textContent =
        data.description ||
        "This result estimates delay risk from the submitted project characteristics.";

    const progress = document.getElementById("resultProgress");
    progress.style.width = `${probability}%`;
    progress.className = `progress-fill ${riskClass(level)}`;
}

function inferRiskLevel(score) {
    if (score >= 80) return "CRITICAL";
    if (score >= 60) return "HIGH";
    if (score >= 30) return "MEDIUM";
    return "LOW";
}

/* Dashboard summary */
async function loadSummary() {
    try {
        const data = await apiGet("/analytics/summary");
        setText("totalProjects", formatNumber(data.total_projects));
        setText("delayedProjects", formatNumber(data.delayed_projects));
        setText("onTimeProjects", formatNumber(data.on_time_projects));
        setText("delayRate", `${Number(data.delay_rate || 0).toFixed(1)}%`);
        setText("analyticsProjectCount", formatNumber(data.total_projects));
        setText("lastUpdated", `Updated ${new Date().toLocaleTimeString()}`);
        setApiStatus(true);
    } catch (error) {
        setApiStatus(false);
        console.error("Summary request failed:", error);
    }
}

/* Analytics and charts */
async function loadAnalytics() {
    try {
        const [
            sectorData,
            yearData,
            delayData,
            costData,
            durationData
        ] = await Promise.all([
            apiGet("/analytics/sector"),
            apiGet("/analytics/year"),
            apiGet("/analytics/delay-distribution"),
            apiGet("/analytics/cost-distribution"),
            apiGet("/analytics/cost-duration")
        ]);

        state.sectorData = Array.isArray(sectorData)
            ? sectorData : [];

        state.yearData = Array.isArray(yearData)
            ? yearData : [];

        renderCharts();
        renderSectorTable();

        renderDelayDistributionChart(delayData);
        renderCostHistogramChart(costData);
        renderCostDurationChart(durationData);

    } catch (error) {
        const table = document.getElementById("sectorTable");

        if (table) {
            table.innerHTML =
                '<tr><td colspan="4">Unable to load analytics.</td></tr>';
        }

        console.error("Analytics request failed:", error);
    }
}

function renderCharts() {
    if (typeof Chart === "undefined") return;

    const sectorCanvas = document.getElementById("sectorChart");
    const yearCanvas = document.getElementById("yearChart");

    if (sectorCanvas) {
        if (state.sectorChart) state.sectorChart.destroy();

        state.sectorChart = new Chart(sectorCanvas, {
            type: "bar",
            data: {
                labels: state.sectorData.map(row => row.sector || "Unknown"),
                datasets: [{
                    label: "Projects",
                    data: state.sectorData.map(row => Number(row.project_count || 0)),
                    backgroundColor: "#3478bd",
                    borderRadius: 5 
                }]
            },
            options: chartOptions()
        });
    }

    if (yearCanvas) {
        if (state.yearChart) state.yearChart.destroy();

        state.yearChart = new Chart(yearCanvas, {
            type: "line",
            data: {
                labels: state.yearData.map(row => String(row.approval_year ?? row.year ?? "Unknown")),
                datasets: [{
                    label: "Projects",
                    data: state.yearData.map(row =>
                        Number(row.project_count ?? row.total_projects ?? row.count ?? 0)
                    ),
                    borderColor: "#3478bd",
                    backgroundColor: "rgba(52,120,189,.12)",
                    fill: true,
                    tension: .3
                }]
            },
            options: chartOptions()
        });
    }
}
function renderDelayDistributionChart(data) {
    const canvas = document.getElementById(
        "delayDistributionChart"
    );

    if (!canvas || typeof Chart === "undefined") return;

    if (state.delayDistributionChart) {
        state.delayDistributionChart.destroy();
    }

    state.delayDistributionChart = new Chart(canvas, {
        type: "pie",
        data: {
            labels: data.labels,
            datasets: [{
                data: data.counts,
                backgroundColor: ["#3478bd", "#8eb8df"],
                borderWidth: 2,
                borderColor: "#ffffff"
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "bottom" },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            const total = context.dataset.data
                                .reduce((sum, value) => sum + value, 0);

                            const count = context.raw;
                            const percent = total
                                ? (count / total * 100).toFixed(1)
                                : "0.0";

                            return `${context.label}: ${count} (${percent}%)`;
                        }
                    }
                }
            }
        }
    });
}


function renderCostHistogramChart(data) {
    const canvas = document.getElementById(
        "costHistogramChart"
    );

    if (!canvas || typeof Chart === "undefined") return;

    if (state.costHistogramChart) {
        state.costHistogramChart.destroy();
    }

    state.costHistogramChart = new Chart(canvas, {
        type: "bar",
        data: {
            labels: data.labels,
            datasets: [{
                label: "Projects",
                data: data.counts,
                backgroundColor: "#3478bd",
                borderRadius: 4
            }]
        },
        options: {
            ...chartOptions(),
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: items =>
                            `Cost range: ${items[0].label}`
                    }
                }
            },
            scales: {
                ...chartOptions().scales,
                x: {
                    ...chartOptions().scales.x,
                    title: {
                        display: true,
                        text: "Original project cost (₹ crore)"
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 0
                    }
                },
                y: {
                    ...chartOptions().scales.y,
                    title: {
                        display: true,
                        text: "Number of projects"
                    }
                }
            }
        }
    });
}


function renderCostDurationChart(data) {
    const canvas = document.getElementById(
        "costDurationChart"
    );

    if (!canvas || typeof Chart === "undefined") return;

    if (state.costDurationChart) {
        state.costDurationChart.destroy();
    }

    const points = Array.isArray(data.points)
        ? data.points : [];

    state.costDurationChart = new Chart(canvas, {
        type: "scatter",
        data: {
            datasets: [
                {
                    label: "Not Delayed",
                    data: points
                        .filter(point => point.delayed === 0)
                        .map(point => ({ x: point.x, y: point.y })),
                    backgroundColor: "#3478bd",
                    pointRadius: 3
                },
                {
                    label: "Delayed",
                    data: points
                        .filter(point => point.delayed === 1)
                        .map(point => ({ x: point.x, y: point.y })),
                    backgroundColor: "#e5a64a",
                    pointRadius: 3
                }
            ]
        },
        options: {
            ...chartOptions(),
            scales: {
                x: {
                    type: "linear",
                    title: {
                        display: true,
                        text: "Original project cost (₹ crore)"
                    },
                    ticks: { maxTicksLimit: 10 },
                    grid: { display: false }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: "Planned duration (months)"
                    },
                    grid: { color: "#edf1f5" }
                }
            }
        }
    });
}

function chartOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "bottom" } },
        scales: {
            x: { grid: { display: false }, ticks: { maxRotation: 40, minRotation: 0 } },
            y: { beginAtZero: true, grid: { color: "#edf1f5" } }
        }
    };
}

function renderSectorTable() {
    const body = document.getElementById("sectorTable");

    if (!state.sectorData.length) {
        body.innerHTML = '<tr><td colspan="4">No sector data available.</td></tr>';
        return;
    }

    body.innerHTML = state.sectorData.map(row => `
        <tr>
            <td>${escapeHTML(row.sector || "Unknown")}</td>
            <td>${formatNumber(row.project_count)}</td>
            <td>${formatNumber(row.delayed_projects)}</td>
            <td>${Number(row.delay_rate || 0).toFixed(1)}%</td>
        </tr>
    `).join("");
}

/* Refresh */
function setupRefreshButtons() {
    document.getElementById("refreshButton")?.addEventListener("click", refreshAll);
}

async function refreshAll() {
    await Promise.allSettled([
        loadOptions(),
        loadSummary(),
        loadAnalytics()
    ]);
    showToast("Dashboard refresh completed.");
}

/* Helpers */
function riskClass(level) {
    return ({
        LOW: "risk-low",
        MEDIUM: "risk-medium",
        HIGH: "risk-high",
        CRITICAL: "risk-critical"
    })[level] || "risk-medium";
}

function setText(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = value ?? "—";
}

function formatNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toLocaleString("en-IN") : "—";
}

function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, character => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
    })[character]);
}

function setApiStatus(online) {
    const element = document.getElementById("apiStatus");
    if (!element) return;

    element.textContent = online ? "Online" : "Offline";
    element.classList.toggle("offline", !online);
}

function showToast(message, isError = false) {
    const toast = document.getElementById("toast");
    if (!toast) return;

    toast.textContent = message;
    toast.classList.toggle("toast-error", isError);
    toast.classList.add("toast-visible");

    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => {
        toast.classList.remove("toast-visible");
    }, 3500);
}
