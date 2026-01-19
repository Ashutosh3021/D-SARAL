// Global variables
let currentStep = 1;
let sessionId = null;
let fileType = 'csv';
let uploadedFiles = [];
let processingInterval = null;

// DOM Elements
const stepElements = {
    1: document.getElementById('step1'),
    2: document.getElementById('step2'),
    3: document.getElementById('step3'),
    4: document.getElementById('step4')
};

const stepIndicators = document.querySelectorAll('.step-indicator');
const fileTypeCards = document.querySelectorAll('.file-type-card');
const nextBtn1 = document.getElementById('nextBtn1');
const prevBtn2 = document.getElementById('prevBtn2');
const processBtn = document.getElementById('processBtn');
const browseBtn = document.getElementById('browseBtn');
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const fileList = document.getElementById('fileList');
const filePreview = document.getElementById('filePreview');
const selectedFileTypeSpan = document.getElementById('selectedFileType');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const statusMessages = document.getElementById('statusMessages');
const downloadBtn = document.getElementById('downloadBtn');
const viewReportBtn = document.getElementById('viewReportBtn');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const reportModal = document.getElementById('reportModal');
const reportContent = document.getElementById('reportContent');
const modalClose = document.querySelector('.modal-close');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    updateStepIndicator(currentStep);
});

// Initialize event listeners
function initializeEventListeners() {
    // File type selection
    fileTypeCards.forEach(card => {
        card.addEventListener('click', function() {
            selectFileType(this.dataset.type);
        });
    });

    // Navigation buttons
    nextBtn1.addEventListener('click', () => goToStep(2));
    prevBtn2.addEventListener('click', () => goToStep(1));
    processBtn.addEventListener('click', processFiles);
    newAnalysisBtn.addEventListener('click', resetWizard);

    // File upload
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelection);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Modal
    modalClose.addEventListener('click', closeModal);
    reportModal.addEventListener('click', function(e) {
        if (e.target === reportModal) {
            closeModal();
        }
    });

    // Download and report buttons
    downloadBtn.addEventListener('click', downloadResults);
    viewReportBtn.addEventListener('click', viewReport);
}

// File type selection
function selectFileType(type) {
    fileType = type;
    
    // Update UI
    fileTypeCards.forEach(card => {
        card.classList.remove('selected');
    });
    
    const selectedCard = document.querySelector(`[data-type="${type}"]`);
    selectedCard.classList.add('selected');
    
    // Update file type display
    const fileTypeNames = {
        'csv': 'CSV',
        'json': 'JSON',
        'txt': 'Text'
    };
    selectedFileTypeSpan.textContent = fileTypeNames[type];
    
    // Enable next button
    nextBtn1.disabled = false;
}

// Navigation between steps
function goToStep(step) {
    // Hide current step
    stepElements[currentStep].classList.remove('active');
    
    // Show new step
    stepElements[step].classList.add('active');
    
    // Update indicators
    updateStepIndicator(step);
    
    currentStep = step;
}

function updateStepIndicator(step) {
    stepIndicators.forEach((indicator, index) => {
        const indicatorStep = index + 1;
        
        if (indicatorStep < step) {
            indicator.classList.add('completed');
            indicator.classList.remove('active');
        } else if (indicatorStep === step) {
            indicator.classList.add('active');
            indicator.classList.remove('completed');
        } else {
            indicator.classList.remove('active', 'completed');
        }
    });
    
    // Update progress lines
    const progressLines = document.querySelectorAll('.progress-line');
    progressLines.forEach((line, index) => {
        if (index < step - 1) {
            line.classList.add('completed');
        } else {
            line.classList.remove('completed');
        }
    });
}

// File upload handling
function handleFileSelection(event) {
    const files = Array.from(event.target.files);
    processSelectedFiles(files);
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const files = Array.from(event.dataTransfer.files);
    processSelectedFiles(files);
}

function processSelectedFiles(files) {
    // Filter files by selected type
    const allowedExtensions = {
        'csv': ['.csv'],
        'json': ['.json'],
        'txt': ['.txt', '.tsv']
    };
    
    const validFiles = files.filter(file => {
        const extension = '.' + file.name.split('.').pop().toLowerCase();
        return allowedExtensions[fileType].includes(extension);
    });
    
    if (validFiles.length === 0) {
        alert(`Please select ${fileType.toUpperCase()} files only.`);
        return;
    }
    
    // Store files and update UI
    uploadedFiles = validFiles;
    updateFileList();
    
    // Enable process button
    processBtn.disabled = false;
}

function updateFileList() {
    fileList.innerHTML = '';
    
    uploadedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div class="file-icon">📄</div>
            <div class="file-info">
                <div class="file-name">${file.name}</div>
                <div class="file-size">${formatFileSize(file.size)}</div>
            </div>
            <button class="remove-file" onclick="removeFile(${index})">&times;</button>
        `;
        fileList.appendChild(fileItem);
    });
    
    filePreview.style.display = uploadedFiles.length > 0 ? 'block' : 'none';
}

function removeFile(index) {
    uploadedFiles.splice(index, 1);
    updateFileList();
    
    if (uploadedFiles.length === 0) {
        processBtn.disabled = true;
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Process files
async function processFiles() {
    if (uploadedFiles.length === 0) {
        alert('Please select files to process.');
        return;
    }
    
    // Go to processing step
    goToStep(3);
    
    try {
        // Upload files
        await uploadFiles();
        
        // Process data
        await processData();
        
    } catch (error) {
        console.error('Processing error:', error);
        showStatusMessage('error', `Processing failed: ${error.message}`);
    }
}

async function uploadFiles() {
    const formData = new FormData();
    
    uploadedFiles.forEach(file => {
        formData.append('files', file);
    });
    formData.append('fileType', fileType);
    
    showStatusMessage('info', 'Uploading files...');
    updateProgress(10, 'Uploading files...');
    
    const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
    }
    
    const result = await response.json();
    sessionId = result.sessionId;
    
    showStatusMessage('success', `Successfully uploaded ${result.files.length} files`);
    updateProgress(25, 'Files uploaded successfully');
}

async function processData() {
    showStatusMessage('info', 'Starting data analysis...');
    updateProgress(30, 'Analyzing data quality...');
    
    // Use Server-Sent Events for real-time updates
    const eventSource = new EventSource(`/api/process/${sessionId}`);
    
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleProcessingUpdate(data);
    };
    
    eventSource.onerror = function(error) {
        console.error('SSE Error:', error);
        eventSource.close();
    };
}

function handleProcessingUpdate(data) {
    updateProgress(data.progress, data.message);
    showStatusMessage('info', data.message);
    
    if (data.status === 'complete') {
        showStatusMessage('success', 'Processing completed!');
        updateProgress(100, 'Complete!');
        setTimeout(() => showResults(data.results), 1000);
    } else if (data.status === 'error') {
        showStatusMessage('error', data.message);
    }
}

function updateProgress(percent, text) {
    progressFill.style.width = `${percent}%`;
    progressText.textContent = text;
}

function showStatusMessage(type, message) {
    const messageElement = document.createElement('div');
    messageElement.className = `status-message ${type}`;
    messageElement.innerHTML = `
        <i class="fas fa-${getStatusIcon(type)}"></i>
        <span>${message}</span>
    `;
    
    statusMessages.appendChild(messageElement);
    
    // Scroll to bottom
    statusMessages.scrollTop = statusMessages.scrollHeight;
    
    // Auto-remove old messages
    if (statusMessages.children.length > 5) {
        statusMessages.removeChild(statusMessages.firstChild);
    }
}

function getStatusIcon(type) {
    const icons = {
        'info': 'info-circle',
        'success': 'check-circle',
        'error': 'exclamation-circle',
        'warning': 'exclamation-triangle'
    };
    return icons[type] || 'info-circle';
}

function showResults(results) {
    // Update statistics
    document.getElementById('filesProcessed').textContent = results.filesProcessed;
    document.getElementById('issuesFound').textContent = results.issuesFound;
    
    // Calculate data quality percentage
    const originalRows = results.originalShape[0];
    const cleanedRows = results.cleanedShape[0];
    const qualityPercentage = originalRows > 0 ? Math.round((cleanedRows / originalRows) * 100) : 100;
    document.getElementById('dataQuality').textContent = `${qualityPercentage}%`;
    document.getElementById('rowsProcessed').textContent = cleanedRows.toLocaleString();
    
    // Update preview
    document.getElementById('previewContent').innerHTML = `<pre>${results.reportPreview}</pre>`;
    
    // Go to results step
    goToStep(4);
}

// Results actions
async function downloadResults() {
    if (!sessionId) {
        alert('No session data available.');
        return;
    }
    
    try {
        const response = await fetch(`/api/download/${sessionId}`);
        if (!response.ok) {
            throw new Error('Download failed');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dsaral_results_${sessionId}.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        showStatusMessage('success', 'Download started!');
    } catch (error) {
        console.error('Download error:', error);
        showStatusMessage('error', 'Download failed. Please try again.');
    }
}

async function viewReport() {
    if (!sessionId) {
        alert('No session data available.');
        return;
    }
    
    try {
        const response = await fetch(`/api/report/${sessionId}`);
        if (!response.ok) {
            throw new Error('Failed to load report');
        }
        
        const data = await response.json();
        reportContent.textContent = data.report;
        reportModal.classList.add('show');
    } catch (error) {
        console.error('Report error:', error);
        showStatusMessage('error', 'Failed to load report.');
    }
}

function closeModal() {
    reportModal.classList.remove('show');
}

// Reset wizard
function resetWizard() {
    // Reset variables
    currentStep = 1;
    sessionId = null;
    fileType = 'csv';
    uploadedFiles = [];
    
    // Reset UI
    fileTypeCards.forEach(card => card.classList.remove('selected'));
    nextBtn1.disabled = true;
    processBtn.disabled = true;
    fileInput.value = '';
    filePreview.style.display = 'none';
    fileList.innerHTML = '';
    
    // Reset progress
    progressFill.style.width = '0%';
    progressText.textContent = 'Initializing...';
    statusMessages.innerHTML = '';
    
    // Go to first step
    goToStep(1);
}

// Cleanup session when leaving page
window.addEventListener('beforeunload', function() {
    if (sessionId) {
        // Clean up server session (best effort)
        fetch(`/api/cleanup/${sessionId}`, { method: 'DELETE' }).catch(() => {});
    }
});