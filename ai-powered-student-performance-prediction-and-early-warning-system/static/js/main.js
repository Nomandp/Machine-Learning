/**
 * Client Interactivity & UI Helper Scripts
 * AI Student Performance Prediction & Early Warning System
 */

document.addEventListener('DOMContentLoaded', function () {
    // Enable Bootstrap Tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Auto dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(function (alert) {
        setTimeout(function () {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 6000);
    });

    // Client-side quick filter for tables
    const quickSearchInput = document.getElementById('tableQuickSearch');
    if (quickSearchInput) {
        quickSearchInput.addEventListener('keyup', function () {
            const term = this.value.toLowerCase();
            const rows = document.querySelectorAll('#studentTableBody tr');
            rows.forEach(function (row) {
                const text = row.innerText.toLowerCase();
                row.style.display = text.includes(term) ? '' : 'none';
            });
        });
    }

    // Batch Re-Predict button handler
    const batchPredictBtn = document.getElementById('btnBatchPredict');
    if (batchPredictBtn) {
        batchPredictBtn.addEventListener('click', function () {
            if (!confirm('Run batch prediction on all students using trained ML models?')) {
                return;
            }
            batchPredictBtn.disabled = true;
            batchPredictBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-1"></i> Processing...';

            fetch('/api/batch-repredict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    alert(`Successfully evaluated ${data.updated_count} students!`);
                    window.location.reload();
                } else {
                    alert('Error: ' + data.error);
                    batchPredictBtn.disabled = false;
                    batchPredictBtn.innerHTML = '<i class="fa-solid fa-arrows-rotate me-1"></i> Re-Predict All';
                }
            })
            .catch(err => {
                alert('Request failed: ' + err);
                batchPredictBtn.disabled = false;
                batchPredictBtn.innerHTML = '<i class="fa-solid fa-arrows-rotate me-1"></i> Re-Predict All';
            });
        });
    }
});
