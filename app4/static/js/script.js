const fileInput = document.getElementById('imageInput');
const previewContainer = document.getElementById('previewContainer');
const uploadLabel = document.querySelector('.file-input-label');
let currentFiles = [];

function updateUploadVisibility() {
    if (currentFiles.length > 0) {
        uploadLabel.style.display = 'none';
    } else {
        uploadLabel.style.display = 'block';
    }
}

function updateFileInput() {
    const dt = new DataTransfer();
    currentFiles.forEach(f => dt.items.add(f));
    fileInput.files = dt.files;
}

function renderPreviews() {
    previewContainer.innerHTML = '';
    currentFiles.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            const previewCard = document.createElement('div');
            previewCard.className = 'preview-card';
            previewCard.innerHTML = `
                <img src="${e.target.result}" alt="Preview ${file.name}">
                <button class="close-btn" title="Remove image">✕</button>
            `;
            previewContainer.appendChild(previewCard);

            previewCard.querySelector('.close-btn').addEventListener('click', function() {
                currentFiles.splice(index, 1);
                updateFileInput();
                renderPreviews();
                updateUploadVisibility();
            });
        };
        reader.readAsDataURL(file);
    });
}

fileInput.addEventListener('change', function(e) {
    currentFiles = Array.from(e.target.files).filter(f => f.type.startsWith('image/'));
    renderPreviews();
    updateUploadVisibility();
});

document.getElementById('uploadForm').addEventListener('submit', function() {
    const classifyBtn = document.getElementById('classifyBtn');
    const skeletonLoader = document.getElementById('skeletonLoader');
    const emptyState = document.querySelector('.empty-state');
    const resultItems = document.querySelectorAll('.result-item');
    
    classifyBtn.classList.add('is-loading');
    skeletonLoader.classList.add('active');
    
    if (emptyState) {
        emptyState.style.display = 'none';
    }
    resultItems.forEach(item => {
        item.style.display = 'none';
    });
});