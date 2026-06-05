document.addEventListener('DOMContentLoaded', () => {
    const trainBtn = document.getElementById('trainBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const dbStatus = document.getElementById('dbStatus');

    trainBtn.addEventListener('click', async () => {
        // Show loading state
        loadingOverlay.classList.remove('hidden');

        try {
            const response = await fetch('/train', {
                method: 'POST',
            });

            const data = await response.json();

            if (response.ok) {
                alert(data.message);
                dbStatus.textContent = "Updated";
                dbStatus.classList.add('active');
            } else {
                alert('Error: ' + data.message);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while training the model.');
        } finally {
            // Hide loading state
            loadingOverlay.classList.add('hidden');
        }
    });

    // Load Dataset Library
    async function loadDataset() {
        const grid = document.getElementById('datasetGrid');
        try {
            const response = await fetch('/dataset');
            const files = await response.json();

            grid.innerHTML = '';

            if (files.length === 0) {
                grid.innerHTML = '<div class="empty-state">No images found</div>';
                return;
            }

            files.forEach(file => {
                const div = document.createElement('div');
                div.className = 'dataset-item';
                div.innerHTML = `<img src="/dataset/${file}" alt="${file}" title="${file}">`;
                grid.appendChild(div);
            });
        } catch (error) {
            console.error('Error loading dataset:', error);
            grid.innerHTML = '<div class="empty-state">Error loading library</div>';
        }
    }

    loadDataset();
});
