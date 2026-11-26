/**
 * Charts for the Download Reports page
 * Uses REPORTS_PAGE_DATA passed from the Flask template
 */

document.addEventListener('DOMContentLoaded', function () {
    if (typeof REPORTS_PAGE_DATA === 'undefined') return;

    const D = REPORTS_PAGE_DATA;

    // ── 1. Risk Distribution Donut ──────────────────────────────
    const riskDonut = document.getElementById('riskDonutChart');
    if (riskDonut) {
        new Chart(riskDonut, {
            type: 'doughnut',
            data: {
                labels: ['At Risk', 'On Track'],
                datasets: [{
                    data: [D.atRisk, D.onTrack],
                    backgroundColor: ['#ef4444', '#10b981'],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '65%',
                plugins: {
                    legend: { position: 'bottom', labels: { font: { size: 12 }, boxWidth: 12 } },
                    tooltip: {
                        callbacks: {
                            label: function (ctx) {
                                const total = D.atRisk + D.onTrack;
                                const pct = total > 0 ? ((ctx.parsed / total) * 100).toFixed(1) : 0;
                                return ` ${ctx.label}: ${ctx.parsed} (${pct}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    // ── 2. Risk Level Bar (High / Moderate / Low) ───────────────
    const riskLevelCanvas = document.getElementById('riskLevelChart');
    if (riskLevelCanvas) {
        new Chart(riskLevelCanvas, {
            type: 'bar',
            data: {
                labels: ['High Risk', 'Moderate', 'Low Risk'],
                datasets: [{
                    label: 'Students',
                    data: [D.highRisk, D.moderateRisk, D.lowRisk],
                    backgroundColor: ['#ef4444', '#f59e0b', '#10b981'],
                    borderRadius: 8,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, ticks: { stepSize: 1 } },
                    x: { grid: { display: false } }
                }
            }
        });
    }

    // ── 3. Cluster / Group Distribution (Polar Area) ────────────
    const clusterCanvas = document.getElementById('clusterChart');
    if (clusterCanvas && D.clusterDist) {
        const clusterLabels = Object.keys(D.clusterDist);
        const clusterValues = Object.values(D.clusterDist);
        new Chart(clusterCanvas, {
            type: 'polarArea',
            data: {
                labels: clusterLabels,
                datasets: [{
                    data: clusterValues,
                    backgroundColor: [
                        'rgba(16,185,129,0.75)',
                        'rgba(245,158,11,0.75)',
                        'rgba(239,68,68,0.75)'
                    ],
                    borderColor: '#fff',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'bottom', labels: { font: { size: 11 }, boxWidth: 12 } }
                }
            }
        });
    }

    // ── 4. G1 vs G2 Grade Bar Chart (all students) ──────────────
    const gradeCanvas = document.getElementById('gradeBarChart');
    if (gradeCanvas && D.students && D.students.length > 0) {
        const labels = D.students.map(s => s.name.split(' ')[0]);
        const g1Data = D.students.map(s => s.g1);
        const g2Data = D.students.map(s => s.g2);

        new Chart(gradeCanvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'G1 Score',
                        data: g1Data,
                        backgroundColor: 'rgba(59,130,246,0.75)',
                        borderColor: '#2563eb',
                        borderWidth: 1,
                        borderRadius: 4
                    },
                    {
                        label: 'G2 Score',
                        data: g2Data,
                        backgroundColor: 'rgba(16,185,129,0.75)',
                        borderColor: '#059669',
                        borderWidth: 1,
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 20,
                        title: { display: true, text: 'Score (out of 20)' }
                    },
                    x: { grid: { display: false } }
                },
                plugins: {
                    legend: { position: 'top', labels: { font: { size: 12 }, boxWidth: 12 } }
                }
            }
        });
    }
});
