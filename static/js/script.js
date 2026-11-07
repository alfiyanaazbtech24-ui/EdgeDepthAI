
"use strict";

// ==========================================
// EdgeDepthAI - Frontend JavaScript
// Flask API: POST /reconstruct
// ==========================================

const API_URL = "/reconstruct";

let selectedDataset = "nyu";

// ==========================================
// DOM ELEMENTS
// ==========================================

const elements = {
    message: document.getElementById("message"),
    runButton: document.getElementById("run-button"),

    rgb: document.getElementById("rgb-image"),
    depth: document.getElementById("depth-image"),
    groundTruth: document.getElementById("gt-image"),

    scale: document.getElementById("scale"),
    method: document.getElementById("method"),

    metrics: document.getElementById("metrics"),
    metricsContent: document.getElementById("metrics-content"),

    processingInfo: document.getElementById("processing-info"),
    processingContent: document.getElementById("processing-content")
};

// ==========================================
// DATASET SELECTION
// ==========================================

function selectDataset(dataset) {

    if (!["nyu", "middlebury"].includes(dataset)) {
        return;
    }

    selectedDataset = dataset;

    document.querySelectorAll(".dataset-card").forEach(card => {

        card.classList.toggle(
            "selected",
            card.dataset.dataset === dataset
        );

    });

    const description = document.getElementById("upload-description");

    if (description) {

        description.textContent =
            dataset === "nyu"
                ? "Upload an NYU Depth V2 RGB image and its corresponding depth map."
                : "Upload a Middlebury RGB image and its corresponding depth map. Your backend must support this dataset.";
    }

    showMessage("");
}

// ==========================================
// MESSAGE HANDLING
// ==========================================

function showMessage(message, type = "info") {

    if (!elements.message) return;

    elements.message.textContent = message;

    elements.message.style.color =
        type === "error" ? "#ff7070" :
        type === "success" ? "#42d9a0" :
        "#b49aff";
}

// ==========================================
// FILE EXTENSION
// ==========================================

function getExtension(filename) {

    return filename.substring(
        filename.lastIndexOf(".")
    ).toLowerCase();
}

// ==========================================
// IMAGE PREVIEW
// ==========================================

function previewImage(file, containerId) {

    const container = document.getElementById(containerId);

    if (!container) return;

    if (!file || !file.type.startsWith("image/")) {
        return;
    }

    const reader = new FileReader();

    reader.onload = event => {

        container.replaceChildren();

        const image = document.createElement("img");

        image.src = event.target.result;
        image.alt = "Selected image preview";

        container.appendChild(image);
    };

    reader.onerror = () => {

        showMessage(
            "Unable to preview the selected image.",
            "error"
        );

    };

    reader.readAsDataURL(file);
}

// ==========================================
// FILE INPUT INITIALIZATION
// ==========================================

function setupFileInput(input, nameElement, previewId = null) {

    if (!input || !nameElement) return;

    input.addEventListener("change", () => {

        const file = input.files[0];

        nameElement.textContent = file
            ? file.name
            : "No file selected";

        if (file && previewId) {
            previewImage(file, previewId);
        }

    });
}

// Initialize all upload fields

setupFileInput(
    elements.rgb,
    document.getElementById("rgb-name"),
    "original-rgb"
);

setupFileInput(
    elements.depth,
    document.getElementById("depth-name"),
    "input-depth"
);

setupFileInput(
    elements.groundTruth,
    document.getElementById("gt-name")
);

// ==========================================
// DISPLAY BACKEND IMAGES
// ==========================================

function displayImage(containerId, imageData, altText) {

    const container = document.getElementById(containerId);

    if (!container || !imageData) return;

    container.replaceChildren();

    const image = document.createElement("img");

    image.src = imageData;
    image.alt = altText;

    image.onerror = () => {

        container.replaceChildren();

        const message = document.createElement("p");

        message.textContent = "Image could not be loaded.";

        container.appendChild(message);
    };

    container.appendChild(image);
}

// ==========================================
// IMAGE VALIDATION
// ==========================================

function validateFiles() {

    const rgb = elements.rgb.files[0];
    const depth = elements.depth.files[0];

    if (!rgb) {
        throw new Error("Please select an RGB image.");
    }

    if (!depth) {
        throw new Error("Please select a depth map.");
    }

    const allowedRGB = [
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff"
    ];

    const allowedDepth = [
        ...allowedRGB,
        ".npy",
        ".pfm"
    ];

    const rgbExtension = getExtension(rgb.name);
    const depthExtension = getExtension(depth.name);

    if (!allowedRGB.includes(rgbExtension)) {
        throw new Error("Unsupported RGB image format.");
    }

    if (!allowedDepth.includes(depthExtension)) {
        throw new Error("Unsupported depth map format.");
    }

    const gt = elements.groundTruth.files[0];

    if (gt) {

        const gtExtension = getExtension(gt.name);

        if (!allowedDepth.includes(gtExtension)) {
            throw new Error("Unsupported ground truth format.");
        }
    }

    return true;
}

// ==========================================
// CREATE FORM DATA
// ==========================================

function createFormData() {

    const formData = new FormData();

    // File inputs

    formData.append("rgb", elements.rgb.files[0]);

    formData.append("depth", elements.depth.files[0]);

    if (elements.groundTruth.files[0]) {

        formData.append(
            "ground_truth",
            elements.groundTruth.files[0]
        );

    }

    // Processing settings

    formData.append("dataset", selectedDataset);

    formData.append("scale", elements.scale.value);

    formData.append("method", elements.method.value);

    // Optional refinement settings

    formData.append("rgb_guidance", "true");

    formData.append("boundary_refinement", "true");

    formData.append("noise_reduction", "none");

    formData.append("hole_filling", "none");

    formData.append("edge_preservation", "enabled");

    return formData;
}

// ==========================================
// MAIN RECONSTRUCTION FUNCTION
// ==========================================

async function runEstimation() {

    if (elements.runButton.disabled) return;

    try {

        // Validate uploaded files

        validateFiles();

        // Deep model is not connected yet

        if (elements.method.value === "deep") {

            throw new Error(
                "Deep inference is not connected yet. Please select Hybrid or another available method."
            );

        }

        const formData = createFormData();

        // Disable button during processing

        elements.runButton.disabled = true;

        elements.runButton.textContent = "Processing...";

        showMessage(
            "Uploading images and running depth reconstruction..."
        );

        // Send request to Flask

        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        // Parse response safely

        const data = await response.json();

        if (!response.ok) {

            throw new Error(
                data.error || "Depth reconstruction failed."
            );

        }

        // Check backend response

        if (!data || typeof data !== "object") {

            throw new Error(
                "Invalid response received from the backend."
            );

        }

        // Display results

        displayResults(data);

        showMessage(
            "Depth reconstruction completed successfully!",
            "success"
        );

    } catch (error) {

        console.error("EdgeDepthAI Error:", error);

        showMessage(
            error.message || "Something went wrong.",
            "error"
        );

    } finally {

        elements.runButton.disabled = false;

        elements.runButton.textContent =
            "Run Reconstruction →";

    }
}

// ==========================================
// DISPLAY ALL RESULTS
// ==========================================

function displayResults(data) {

    // Original RGB image is already displayed
    // during file selection.

    // Original/input depth visualization

    displayImage(
        "input-depth",
        data.original,
        "Input Depth Map"
    );

    // Reconstructed depth map

    displayImage(
        "refined-depth",
        data.refined,
        "Reconstructed Depth Map"
    );

    // Edge detection

    displayImage(
        "edge-result",
        data.edges,
        "Edge Detection"
    );

    // Boundary refinement

    displayImage(
        "boundary-result",
        data.boundary,
        "Boundary Refinement"
    );

    // Additional information

    displayMetrics(data);

    displayProcessingInfo(data);
}

// ==========================================
// DISPLAY EVALUATION METRICS
// ==========================================

function displayMetrics(data) {

    const values = [
        data.mae,
        data.rmse,
        data.psnr,
        data.ssim
    ];

    // Metrics require ground truth

    if (
        values.some(value =>
            value === null ||
            value === undefined ||
            !Number.isFinite(Number(value))
        )
    ) {

        elements.metrics.classList.add("hidden");

        return;
    }

    const metrics = {
        MAE: data.mae,
        RMSE: data.rmse,
        PSNR: data.psnr,
        SSIM: data.ssim
    };

    elements.metricsContent.replaceChildren();

    Object.entries(metrics).forEach(([name, value]) => {

        const item = document.createElement("div");

        item.className = "metric-item";

        const label = document.createElement("span");

        label.textContent = name;

        const result = document.createElement("h3");

        result.textContent = Number(value).toFixed(4);

        item.append(label, result);

        elements.metricsContent.appendChild(item);

    });

    elements.metrics.classList.remove("hidden");
}

// ==========================================
// DISPLAY PROCESSING INFORMATION
// ==========================================

function displayProcessingInfo(data) {

    elements.processingContent.replaceChildren();

    const information = [

        [
            "Dataset",
            selectedDataset === "nyu"
                ? "NYU Depth V2"
                : "Middlebury Stereo"
        ],

        ["Method", data.method],

        ["Scale", data.scale ? `${data.scale}x` : null],

        ["Output Resolution", data.output_resolution],

        [
            "Processing Time",
            data.processing_time != null
                ? `${data.processing_time} ms`
                : null
        ],

        ["Valid Pixels", data.valid_pixels],

        ["Memory Usage", data.memory_usage ?? "Not available"]

    ];

    information.forEach(([label, value]) => {

        const line = document.createElement("p");

        line.textContent = `${label}: ${value ?? "N/A"}`;

        elements.processingContent.appendChild(line);

    });

    elements.processingInfo.classList.remove("hidden");
}

// ==========================================
// RESET PLACEHOLDER
// ==========================================

function setPlaceholder(containerId, icon, message) {

    const container = document.getElementById(containerId);

    if (!container) return;

    container.replaceChildren();

    const iconElement = document.createElement("span");

    iconElement.textContent = icon;

    const textElement = document.createElement("p");

    textElement.textContent = message;

    container.append(iconElement, textElement);
}

// ==========================================
// RESET FORM
// ==========================================

function resetForm() {

    // Clear file inputs

    [
        elements.rgb,
        elements.depth,
        elements.groundTruth
    ].forEach(input => {

        if (input) {
            input.value = "";
        }

    });

    // Reset filenames

    const filenames = [
        ["rgb-name", "No image selected"],
        ["depth-name", "No depth map selected"],
        ["gt-name", "No ground truth selected"]
    ];

    filenames.forEach(([id, message]) => {

        const element = document.getElementById(id);

        if (element) {
            element.textContent = message;
        }

    });

    // Reset result images

    setPlaceholder(
        "original-rgb",
        "◫",
        "Waiting for input"
    );

    setPlaceholder(
        "input-depth",
        "◈",
        "Waiting for input"
    );

    setPlaceholder(
        "refined-depth",
        "◈",
        "Waiting for reconstruction"
    );

    setPlaceholder(
        "edge-result",
        "⌁",
        "Waiting for processing"
    );

    setPlaceholder(
        "boundary-result",
        "◇",
        "Waiting for processing"
    );

    // Hide information panels

    elements.metrics.classList.add("hidden");

    elements.processingInfo.classList.add("hidden");

    // Clear message

    showMessage("");
}

// ==========================================
// INITIALIZATION
// ==========================================

selectDataset("nyu");
