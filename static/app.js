(() => {
    "use strict";

    const CONFIG = Object.freeze({
        uploadEndpoint: "/predict/upload",
        healthEndpoint: "/health",
        readyEndpoint: "/health/ready",

        maxUploadSizeBytes:
            25 * 1024 * 1024,

        allowedExtensions: new Set([
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp"
        ]),

        healthIntervalMs: 30000
    });

    const state = {
        selectedFile: null,
        result: null,
        objectUrl: null,
        healthTimer: null,
        isProcessing: false
    };

    const $ = (id) =>
        document.getElementById(id);

    const elements = Object.freeze({

        serviceStatus:
            $("serviceStatus"),

        dropZone:
            $("dropZone"),

        chooseFileButton:
            $("chooseFileButton"),

        fileInput:
            $("fileInput"),

        selectedFile:
            $("selectedFile"),

        previewImage:
            $("previewImage"),

        fileName:
            $("fileName"),

        fileMeta:
            $("fileMeta"),

        removeFileButton:
            $("removeFileButton"),

        processButton:
            $("processButton"),

        progressSection:
            $("progressSection"),

        progressText:
            $("progressText"),

        progressPercent:
            $("progressPercent"),

        progressBar:
            $("progressBar"),

        errorAlert:
            $("errorAlert"),

        refreshHealthButton:
            $("refreshHealthButton"),

        healthRing:
            $("healthRing"),

        healthRingText:
            $("healthRingText"),

        healthStatus:
            $("healthStatus"),

        healthDescription:
            $("healthDescription"),

        pipelineReady:
            $("pipelineReady"),

        modelLoaded:
            $("modelLoaded"),

        apiVersion:
            $("apiVersion"),

        modelPath:
            $("modelPath"),

        emptyState:
            $("emptyState"),

        resultContent:
            $("resultContent"),

        copyJsonButton:
            $("copyJsonButton"),

        downloadJsonButton:
            $("downloadJsonButton"),

        resultReceiptId:
            $("resultReceiptId"),

        resultProcessingTime:
            $("resultProcessingTime"),

        resultReliability:
            $("resultReliability"),

        resultReview:
            $("resultReview"),

        fieldCountBadge:
            $("fieldCountBadge"),

        fieldsContainer:
            $("fieldsContainer"),

        jsonViewer:
            $("jsonViewer"),

        qualityOcr:
            $("qualityOcr"),

        qualityPattern:
            $("qualityPattern"),

        qualityContext:
            $("qualityContext"),

        qualityOverall:
            $("qualityOverall"),

        financialItems:
            $("financialItems"),

        financialExpectedItems:
            $("financialExpectedItems"),

        financialItemSum:
            $("financialItemSum"),

        financialTotal:
            $("financialTotal"),

        toastContainer:
            $("toastContainer")
    });

    function utcNow() {
        return new Date().toISOString();
    }

    function formatBytes(bytes) {

        if (
            !Number.isFinite(bytes) ||
            bytes < 0
        ) {
            return "—";
        }

        if (bytes < 1024) {
            return `${bytes} B`;
        }

        const units = [
            "KB",
            "MB",
            "GB"
        ];

        let value =
            bytes / 1024;

        let unitIndex = 0;

        while (
            value >= 1024 &&
            unitIndex <
                units.length - 1
        ) {
            value /= 1024;
            unitIndex += 1;
        }

        const decimals =
            value >= 100
                ? 0
                : 1;

        return (
            `${value.toFixed(decimals)} ` +
            `${units[unitIndex]}`
        );
    }

    function safeNumber(
        value,
        fallback = 0
    ) {

        const numeric =
            Number(value);

        return Number.isFinite(numeric)
            ? numeric
            : fallback;
    }

    function normalizeConfidence(
        value
    ) {

        const numeric =
            Number(value);

        if (
            !Number.isFinite(numeric)
        ) {
            return null;
        }

        const normalized =
            numeric > 1
                ? numeric / 100
                : numeric;

        return Math.max(
            0,
            Math.min(
                1,
                normalized
            )
        );
    }

    function formatConfidence(
        value
    ) {

        const normalized =
            normalizeConfidence(
                value
            );

        if (
            normalized === null
        ) {
            return "—";
        }

        return (
            `${(
                normalized * 100
            ).toFixed(1)}%`
        );
    }

    function formatCurrency(
        value
    ) {

        const numeric =
            Number(value);

        if (
            !Number.isFinite(numeric)
        ) {
            return "—";
        }

        return numeric.toFixed(2);
    }

    function displayValue(
        value
    ) {

        if (
            value === null ||
            value === undefined ||
            value === ""
        ) {
            return "—";
        }

        if (
            typeof value === "object"
        ) {
            return JSON.stringify(
                value
            );
        }

        return String(value);
    }

    function getExtension(
        filename
    ) {

        const cleanName =
            String(
                filename || ""
            ).split("?")[0];

        const lastDot =
            cleanName.lastIndexOf(
                "."
            );

        if (lastDot < 0) {
            return "";
        }

        return cleanName
            .slice(lastDot)
            .toLowerCase();
    }

    function getReliabilityClass(
        value
    ) {

        const normalized =
            String(
                value || ""
            ).toLowerCase();

        if (
            normalized === "high"
        ) {
            return "confidence-high";
        }

        if (
            normalized === "medium"
        ) {
            return "confidence-medium";
        }

        if (
            normalized === "low"
        ) {
            return "confidence-low";
        }

        return "confidence-unknown";
    }

    function showToast(
        message,
        type = "success",
        durationMs = 3500
    ) {

        const toast =
            document.createElement(
                "div"
            );

        toast.className =
            `toast toast-${type}`;

        toast.textContent =
            String(message);

        elements.toastContainer
            .appendChild(
                toast
            );

        window.setTimeout(
            () => {
                toast.remove();
            },
            durationMs
        );
    }

    function showError(
        message
    ) {

        elements.errorAlert.textContent =
            String(message);

        elements.errorAlert.classList.remove(
            "hidden"
        );
    }

    function clearError() {

        elements.errorAlert.textContent =
            "";

        elements.errorAlert.classList.add(
            "hidden"
        );
    }

    function setProgress(
        percent,
        text
    ) {

        const safePercent =
            Math.max(
                0,
                Math.min(
                    100,
                    Number(percent) || 0
                )
            );

        elements.progressSection
            .classList.remove(
                "hidden"
            );

        elements.progressPercent
            .textContent =
            `${Math.round(safePercent)}%`;

        elements.progressText
            .textContent =
            text ||
            "Processing receipt...";

        elements.progressBar.style.width =
            `${safePercent}%`;
    }

    function resetProgress() {

        elements.progressSection
            .classList.add(
                "hidden"
            );

        elements.progressBar.style.width =
            "0%";

        elements.progressPercent
            .textContent =
            "0%";

        elements.progressText
            .textContent =
            "Processing receipt...";
    }

    function setProcessButtonState() {

        elements.processButton.disabled =
            !state.selectedFile ||
            state.isProcessing;
    }

    function revokeObjectUrl() {

        if (state.objectUrl) {

            URL.revokeObjectURL(
                state.objectUrl
            );

            state.objectUrl = null;
        }
    }

    function clearSelectedFile() {

        revokeObjectUrl();

        state.selectedFile =
            null;

        elements.fileInput.value =
            "";

        elements.selectedFile
            .classList.add(
                "hidden"
            );

        elements.previewImage
            .removeAttribute(
                "src"
            );

        elements.fileName.textContent =
            "No file selected";

        elements.fileMeta.textContent =
            "—";

        setProcessButtonState();
    }

    function validateFile(
        file
    ) {

        if (!file) {
            throw new Error(
                "Please select a receipt image."
            );
        }

        if (file.size <= 0) {
            throw new Error(
                "The selected file is empty."
            );
        }

        if (
            file.size >
            CONFIG.maxUploadSizeBytes
        ) {
            throw new Error(
                "The selected file exceeds the 25 MB upload limit."
            );
        }

        const extension =
            getExtension(
                file.name
            );

        if (
            !CONFIG.allowedExtensions.has(
                extension
            )
        ) {
            throw new Error(
                "Unsupported image format. " +
                "Use JPG, JPEG, PNG, BMP, " +
                "TIFF, or WEBP."
            );
        }

        return true;
    }

    function selectFile(
        file
    ) {

        clearError();

        try {

            validateFile(
                file
            );

        } catch (error) {

            showError(
                error.message
            );

            showToast(
                error.message,
                "error"
            );

            return;
        }

        revokeObjectUrl();

        state.selectedFile =
            file;

        state.objectUrl =
            URL.createObjectURL(
                file
            );

        elements.previewImage.src =
            state.objectUrl;

        elements.fileName.textContent =
            file.name;

        elements.fileMeta.textContent =
            `${formatBytes(file.size)} · ` +
            `${file.type || getExtension(file.name).toUpperCase()}`;

        elements.selectedFile
            .classList.remove(
                "hidden"
            );

        setProcessButtonState();

        showToast(
            "Receipt selected and ready for extraction.",
            "success"
        );
    }

    async function parseResponse(
        response
    ) {

        const contentType =
            response.headers.get(
                "content-type"
            ) || "";

        if (
            contentType.includes(
                "application/json"
            )
        ) {
            return response.json();
        }

        const text =
            await response.text();

        return {
            success:
                response.ok,
            detail:
                text
        };
    }

    async function fetchJson(
        url,
        options = {}
    ) {

        const response =
            await fetch(
                url,
                {
                    ...options,

                    headers: {
                        Accept:
                            "application/json",

                        ...(
                            options.headers ||
                            {}
                        )
                    }
                }
            );

        const payload =
            await parseResponse(
                response
            );

        if (!response.ok) {

            const detail =
                payload?.detail ||
                payload?.message ||
                "Request failed.";

            throw new Error(
                typeof detail === "string"
                    ? detail
                    : JSON.stringify(
                        detail
                    )
            );
        }

        return payload;
    }

    function setServiceStatus(
        status
    ) {

        const dot =
            elements.serviceStatus
                .querySelector(
                    ".status-dot"
                );

        const text =
            elements.serviceStatus
                .querySelector(
                    ".status-text"
                );

        dot.className =
            "status-dot";

        if (
            status === "healthy"
        ) {

            dot.classList.add(
                "status-ok"
            );

            text.textContent =
                "Service healthy";

            return;
        }

        if (
            status === "degraded"
        ) {

            dot.classList.add(
                "status-warn"
            );

            text.textContent =
                "Service degraded";

            return;
        }

        if (
            status === "unhealthy"
        ) {

            dot.classList.add(
                "status-error"
            );

            text.textContent =
                "Service unavailable";

            return;
        }

        dot.classList.add(
            "status-unknown"
        );

        text.textContent =
            "Service status unknown";
    }

    function updateHealthUI(
        payload
    ) {

        const status =
            String(
                payload?.status ||
                "unknown"
            ).toLowerCase();

        const isHealthy =
            status === "healthy";

        const isDegraded =
            status === "degraded";

        setServiceStatus(
            status
        );

        elements.healthRing.className =
            "health-ring";

        if (isHealthy) {

            elements.healthRing
                .classList.add(
                    "health-ok"
                );

            elements.healthRingText
                .textContent =
                "Ready";

            elements.healthStatus
                .textContent =
                "Healthy";

            elements.healthDescription
                .textContent =
                "Inference pipeline is ready to process receipts.";

        } else if (isDegraded) {

            elements.healthRing
                .classList.add(
                    "health-warn"
                );

            elements.healthRingText
                .textContent =
                "Degraded";

            elements.healthStatus
                .textContent =
                "Degraded";

            elements.healthDescription
                .textContent =
                "Service is running but one or more readiness checks need attention.";

        } else {

            elements.healthRing
                .classList.add(
                    "health-error"
                );

            elements.healthRingText
                .textContent =
                "Down";

            elements.healthStatus
                .textContent =
                "Unavailable";

            elements.healthDescription
                .textContent =
                "The inference pipeline is not currently ready.";
        }

        elements.pipelineReady
            .textContent =
            payload?.pipeline_ready
                ? "Ready"
                : "Not ready";

        elements.modelLoaded
            .textContent =
            payload?.model_loaded
                ? "Loaded"
                : "Not loaded";

        elements.apiVersion
            .textContent =
            payload?.version ||
            "—";
    }

    async function refreshHealth() {

        try {

            const payload =
                await fetchJson(
                    CONFIG.healthEndpoint
                );

            updateHealthUI(
                payload
            );

            if (
                payload?.model_path
            ) {

                elements.modelPath
                    .textContent =
                    payload.model_path;

                elements.modelPath.title =
                    payload.model_path;

            } else {

                elements.modelPath
                    .textContent =
                    "Not reported";

                elements.modelPath.title =
                    "";
            }

            return payload;

        } catch (error) {

            setServiceStatus(
                "unhealthy"
            );

            elements.healthRing.className =
                "health-ring health-error";

            elements.healthRingText
                .textContent =
                "Error";

            elements.healthStatus
                .textContent =
                "Unavailable";

            elements.healthDescription
                .textContent =
                error.message;

            elements.pipelineReady
                .textContent =
                "—";

            elements.modelLoaded
                .textContent =
                "—";

            elements.apiVersion
                .textContent =
                "—";

            elements.modelPath
                .textContent =
                "—";

            throw error;
        }
    }

    async function checkReadiness() {

        return fetchJson(
            CONFIG.readyEndpoint
        );
    }

    function findPredictionObject(
        payload
    ) {

        if (
            payload &&
            typeof payload.prediction ===
                "object" &&
            payload.prediction !== null
        ) {
            return payload.prediction;
        }

        return payload || {};
    }

    function findFieldContainer(
        source
    ) {

        const candidates = [
            source?.fields,
            source?.field_confidence,
            source?.extracted_fields
        ];

        for (
            const candidate
            of candidates
        ) {

            if (
                candidate &&
                typeof candidate ===
                    "object"
            ) {
                return candidate;
            }
        }

        return {};
    }

    function normalizeFields(
        prediction
    ) {

        const fieldContainer =
            findFieldContainer(
                prediction
            );

        const fields = [];

        if (
            Array.isArray(
                fieldContainer
            )
        ) {

            for (
                const item
                of fieldContainer
            ) {

                if (
                    !item ||
                    typeof item !==
                        "object"
                ) {
                    continue;
                }

                fields.push({

                    name:
                        item.field_name ||
                        item.name ||
                        "field",

                    value:
                        item.value,

                    confidence:
                        normalizeConfidence(
                            item.confidence
                        ),

                    reliability:
                        item.reliability,

                    ocr_confidence:
                        item.ocr_confidence,

                    pattern_confidence:
                        item.pattern_confidence,

                    context_confidence:
                        item.context_confidence

                });
            }

            return fields;
        }

        for (
            const [
                name,
                rawValue
            ]
            of Object.entries(
                fieldContainer
            )
        ) {

            if (
                rawValue &&
                typeof rawValue ===
                    "object" &&
                !Array.isArray(
                    rawValue
                ) &&
                (
                    Object.prototype
                        .hasOwnProperty.call(
                            rawValue,
                            "value"
                        ) ||
                    Object.prototype
                        .hasOwnProperty.call(
                            rawValue,
                            "confidence"
                        )
                )
            ) {

                fields.push({

                    name,

                    value:
                        rawValue.value,

                    confidence:
                        normalizeConfidence(
                            rawValue.confidence
                        ),

                    reliability:
                        rawValue.reliability,

                    ocr_confidence:
                        rawValue.ocr_confidence,

                    pattern_confidence:
                        rawValue.pattern_confidence,

                    context_confidence:
                        rawValue.context_confidence

                });

            } else {

                fields.push({

                    name,

                    value:
                        rawValue,

                    confidence:
                        null,

                    reliability:
                        null,

                    ocr_confidence:
                        null,

                    pattern_confidence:
                        null,

                    context_confidence:
                        null
                });
            }
        }

        return fields;
    }

    function renderFieldList(
        prediction
    ) {

        const fields =
            normalizeFields(
                prediction
            );

        elements.fieldCountBadge
            .textContent =
            `${fields.length} field${fields.length === 1 ? "" : "s"}`;

        elements.fieldsContainer
            .replaceChildren();

        if (
            !fields.length
        ) {

            const empty =
                document.createElement(
                    "div"
                );

            empty.className =
                "muted";

            empty.textContent =
                "No field-confidence records were returned.";

            elements.fieldsContainer
                .appendChild(
                    empty
                );

            return;
        }

        const fragment =
            document.createDocumentFragment();

        for (
            const field
            of fields
        ) {

            const row =
                document.createElement(
                    "div"
                );

            row.className =
                "field-item";

            const name =
                document.createElement(
                    "div"
                );

            name.className =
                "field-name";

            name.textContent =
                field.name;

            const value =
                document.createElement(
                    "div"
                );

            value.className =
                "field-value";

            value.textContent =
                displayValue(
                    field.value
                );

            const pill =
                document.createElement(
                    "span"
                );

            const confidenceClass =
                getReliabilityClass(
                    field.reliability
                );

            pill.className =
                `confidence-pill ${confidenceClass}`;

            if (
                field.confidence !== null
            ) {

                pill.textContent =
                    formatConfidence(
                        field.confidence
                    );

                pill.title =
                    field.reliability
                        ? (
                            `Reliability: ` +
                            `${field.reliability}`
                        )
                        : "Field confidence";

            } else {

                pill.textContent =
                    field.reliability ||
                    "—";
            }

            row.appendChild(name);
            row.appendChild(value);
            row.appendChild(pill);

            fragment.appendChild(
                row
            );
        }

        elements.fieldsContainer
            .appendChild(
                fragment
            );
    }

    function getFirstDefined(
        object,
        keys
    ) {

        for (
            const key
            of keys
        ) {

            if (
                object &&
                Object.prototype
                    .hasOwnProperty.call(
                        object,
                        key
                    ) &&
                object[key] !== null &&
                object[key] !== undefined
            ) {

                return object[key];
            }
        }

        return undefined;
    }

    function renderQualitySignals(
        prediction
    ) {

        const overall =
            getFirstDefined(
                prediction,
                [
                    "overall_confidence",
                    "confidence"
                ]
            );

        const directOcr =
            getFirstDefined(
                prediction,
                [
                    "ocr_confidence"
                ]
            );

        const directPattern =
            getFirstDefined(
                prediction,
                [
                    "pattern_confidence"
                ]
            );

        const directContext =
            getFirstDefined(
                prediction,
                [
                    "context_confidence"
                ]
            );

        const fields =
            normalizeFields(
                prediction
            );

        const average = (
            key
        ) => {

            const values =
                fields
                    .map(
                        field =>
                            normalizeConfidence(
                                field[key]
                            )
                    )
                    .filter(
                        value =>
                            value !== null
                    );

            if (
                !values.length
            ) {
                return null;
            }

            return (
                values.reduce(
                    (
                        sum,
                        value
                    ) =>
                        sum + value,
                    0
                ) / values.length
            );
        };

        const ocrAverage =
            average(
                "ocr_confidence"
            );

        const patternAverage =
            average(
                "pattern_confidence"
            );

        const contextAverage =
            average(
                "context_confidence"
            );

        elements.qualityOcr
            .textContent =
            directOcr !== undefined
                ? formatConfidence(
                    directOcr
                )
                : ocrAverage !== null
                    ? formatConfidence(
                        ocrAverage
                    )
                    : "—";

        elements.qualityPattern
            .textContent =
            directPattern !== undefined
                ? formatConfidence(
                    directPattern
                )
                : patternAverage !== null
                    ? formatConfidence(
                        patternAverage
                    )
                    : "—";

        elements.qualityContext
            .textContent =
            directContext !== undefined
                ? formatConfidence(
                    directContext
                )
                : contextAverage !== null
                    ? formatConfidence(
                        contextAverage
                    )
                    : "—";

        elements.qualityOverall
            .textContent =
            overall !== undefined
                ? formatConfidence(
                    overall
                )
                : "—";
    }

    function findItems(
        prediction
    ) {

        const items =
            getFirstDefined(
                prediction,
                [
                    "items",
                    "line_items",
                    "extracted_items"
                ]
            );

        return Array.isArray(items)
            ? items
            : [];
    }

    function renderFinancialSummary(
        prediction
    ) {

        const items =
            findItems(
                prediction
            );

        const expectedItems =
            getFirstDefined(
                prediction,
                [
                    "expected_item_count"
                ]
            );

        const itemSum =
            getFirstDefined(
                prediction,
                [
                    "item_sum",
                    "items_total",
                    "subtotal"
                ]
            );

        const total =
            getFirstDefined(
                prediction,
                [
                    "total_amount",
                    "grand_total",
                    "total"
                ]
            );

        const itemCount =
            getFirstDefined(
                prediction,
                [
                    "item_count"
                ]
            );

        elements.financialItems
            .textContent =
            items.length
                ? items.length
                : itemCount ??
                  "0";

        elements.financialExpectedItems
            .textContent =
            expectedItems ??
            "—";

        elements.financialItemSum
            .textContent =
            itemSum !== undefined
                ? formatCurrency(
                    itemSum
                )
                : "—";

        elements.financialTotal
            .textContent =
            total !== undefined
                ? formatCurrency(
                    total
                )
                : "—";
    }

    function renderResult(
        response
    ) {

        const prediction =
            findPredictionObject(
                response
            );

        state.result = {
            ...response,
            prediction
        };

        elements.emptyState
            .classList.add(
                "hidden"
            );

        elements.resultContent
            .classList.remove(
                "hidden"
            );

        elements.copyJsonButton.disabled =
            false;

        elements.downloadJsonButton.disabled =
            false;

        const receiptId =
            response?.receipt_id ||
            prediction?.receipt_id ||
            prediction?.id ||
            state.selectedFile?.name ||
            "—";

        elements.resultReceiptId
            .textContent =
            displayValue(
                receiptId
            );

        if (
            response?.processing_time_ms !==
            undefined
        ) {

            elements.resultProcessingTime
                .textContent =
                `${safeNumber(
                    response.processing_time_ms,
                    0
                ).toFixed(2)} ms`;

        } else {

            elements.resultProcessingTime
                .textContent =
                "—";
        }

        const reliability =
            getFirstDefined(
                prediction,
                [
                    "reliability",
                    "overall_reliability"
                ]
            );

        elements.resultReliability
            .textContent =
            reliability ||
            "—";

        const reviewRequired =
            getFirstDefined(
                prediction,
                [
                    "review_required"
                ]
            );

        elements.resultReview
            .textContent =
            typeof reviewRequired ===
                "boolean"
                ? (
                    reviewRequired
                        ? "Yes"
                        : "No"
                )
                : "—";

        renderFieldList(
            prediction
        );

        renderQualitySignals(
            prediction
        );

        renderFinancialSummary(
            prediction
        );

        elements.jsonViewer
            .textContent =
            JSON.stringify(
                response,
                null,
                2
            );
    }

    async function copyJson() {

        if (
            !state.result
        ) {
            return;
        }

        const content =
            JSON.stringify(
                state.result,
                null,
                2
            );

        try {

            await navigator
                .clipboard
                .writeText(
                    content
                );

            showToast(
                "JSON copied to clipboard.",
                "success"
            );

        } catch (error) {

            showToast(
                "Clipboard access is unavailable in this browser.",
                "warning"
            );
        }
    }

    function sanitizeFilename(
        value
    ) {

        return String(
            value ||
            "receipt-prediction"
        )
            .replace(
                /[^a-z0-9._-]+/gi,
                "_"
            )
            .slice(
                0,
                120
            );
    }

    function downloadJson() {

        if (
            !state.result
        ) {
            return;
        }

        const blob =
            new Blob(
                [
                    JSON.stringify(
                        state.result,
                        null,
                        2
                    )
                ],
                {
                    type:
                        "application/json"
                }
            );

        const url =
            URL.createObjectURL(
                blob
            );

        const receiptId =
            state.result.receipt_id ||
            state.result.prediction?.receipt_id ||
            state.selectedFile?.name ||
            "receipt-prediction";

        const anchor =
            document.createElement(
                "a"
            );

        anchor.href =
            url;

        anchor.download =
            `${sanitizeFilename(
                receiptId
            )}.json`;

        document.body.appendChild(
            anchor
        );

        anchor.click();

        anchor.remove();

        URL.revokeObjectURL(
            url
        );

        showToast(
            "Prediction JSON downloaded.",
            "success"
        );
    }

    async function processReceipt() {

        if (
            !state.selectedFile ||
            state.isProcessing
        ) {
            return;
        }

        state.isProcessing =
            true;

        clearError();

        setProcessButtonState();

        elements.processButton
            .classList.add(
                "is-loading"
            );

        elements.processButton
            .textContent =
            "Processing...";

        setProgress(
            8,
            "Checking service readiness..."
        );

        try {

            await checkReadiness();

            setProgress(
                24,
                "Uploading receipt..."
            );

            const formData =
                new FormData();

            formData.append(
                "file",
                state.selectedFile
            );

            setProgress(
                42,
                "Running receipt inference..."
            );

            const response =
                await fetch(
                    CONFIG.uploadEndpoint,
                    {
                        method:
                            "POST",

                        body:
                            formData,

                        headers: {
                            Accept:
                                "application/json"
                        }
                    }
                );

            setProgress(
                72,
                "Validating structured output..."
            );

            const payload =
                await parseResponse(
                    response
                );

            if (
                !response.ok
            ) {

                const detail =
                    payload?.detail ||
                    payload?.message ||
                    "Receipt inference failed.";

                throw new Error(
                    typeof detail ===
                        "string"
                        ? detail
                        : JSON.stringify(
                            detail
                        )
                );
            }

            setProgress(
                100,
                "Extraction completed."
            );

            renderResult(
                payload
            );

            showToast(
                "Receipt processed successfully.",
                "success"
            );

            window.setTimeout(
                resetProgress,
                600
            );

        } catch (error) {

            setProgress(
                100,
                "Processing failed."
            );

            const message =
                error?.message ||
                "Receipt processing failed.";

            showError(
                message
            );

            showToast(
                message,
                "error"
            );

        } finally {

            state.isProcessing =
                false;

            elements.processButton
                .classList.remove(
                    "is-loading"
                );

            elements.processButton
                .textContent =
                "Run extraction";

            setProcessButtonState();
        }
    }

    function wireDropZone() {

        const preventDefaults =
            (event) => {

                event.preventDefault();
                event.stopPropagation();
            };

        [
            "dragenter",
            "dragover",
            "dragleave",
            "drop"
        ].forEach(
            eventName => {

                elements.dropZone
                    .addEventListener(
                        eventName,
                        preventDefaults
                    );
            }
        );

        [
            "dragenter",
            "dragover"
        ].forEach(
            eventName => {

                elements.dropZone
                    .addEventListener(
                        eventName,
                        () => {

                            elements.dropZone
                                .classList.add(
                                    "dragover"
                                );
                        }
                    );
            }
        );

        [
            "dragleave",
            "drop"
        ].forEach(
            eventName => {

                elements.dropZone
                    .addEventListener(
                        eventName,
                        () => {

                            elements.dropZone
                                .classList.remove(
                                    "dragover"
                                );
                        }
                    );
            }
        );

        elements.dropZone.addEventListener(
            "drop",
            event => {

                const files =
                    event.dataTransfer?.files;

                if (
                    files &&
                    files.length
                ) {

                    selectFile(
                        files[0]
                    );
                }
            }
        );

        elements.dropZone.addEventListener(
            "keydown",
            event => {

                if (
                    event.key === "Enter" ||
                    event.key === " "
                ) {

                    event.preventDefault();

                    elements.fileInput.click();
                }
            }
        );
    }

    function initialize() {

        elements.chooseFileButton
            .addEventListener(
                "click",
                () => {

                    elements.fileInput.click();
                }
            );

        elements.dropZone
            .addEventListener(
                "click",
                event => {

                    if (
                        event.target ===
                        elements.chooseFileButton
                    ) {
                        return;
                    }

                    elements.fileInput.click();
                }
            );

        elements.fileInput
            .addEventListener(
                "change",
                () => {

                    selectFile(
                        elements.fileInput.files?.[0]
                    );
                }
            );

        elements.removeFileButton
            .addEventListener(
                "click",
                clearSelectedFile
            );

        elements.processButton
            .addEventListener(
                "click",
                processReceipt
            );

        elements.refreshHealthButton
            .addEventListener(
                "click",
                async () => {

                    try {

                        elements.refreshHealthButton
                            .disabled =
                            true;

                        await refreshHealth();

                        showToast(
                            "Service health refreshed.",
                            "success"
                        );

                    } catch (error) {

                        showToast(
                            "Unable to refresh service health.",
                            "error"
                        );

                    } finally {

                        elements.refreshHealthButton
                            .disabled =
                            false;
                    }
                }
            );

        elements.copyJsonButton
            .addEventListener(
                "click",
                copyJson
            );

        elements.downloadJsonButton
            .addEventListener(
                "click",
                downloadJson
            );

        wireDropZone();

        refreshHealth()
            .catch(
                () => {}
            );

        state.healthTimer =
            window.setInterval(
                () => {

                    refreshHealth()
                        .catch(
                            () => {}
                        );
                },
                CONFIG.healthIntervalMs
            );

        setProcessButtonState();
    }

    if (
        document.readyState ===
        "loading"
    ) {

        document.addEventListener(
            "DOMContentLoaded",
            initialize,
            {
                once: true
            }
        );

    } else {

        initialize();
    }
})();