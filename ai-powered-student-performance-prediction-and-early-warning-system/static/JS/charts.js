/**
 * Interactive Visualizations using Chart.js 4.4
 * AI-Powered Student Performance Prediction & Early Warning System
 */

document.addEventListener('DOMContentLoaded', function () {

    // -------------------------------------------------------------
    // DASHBOARD CHARTS
    // -------------------------------------------------------------
    if (typeof DASHBOARD_DATA !== 'undefined') {

        // 1. Risk Status Donut Chart
        const riskCanvas = document.getElementById('riskDistributionChart');
        if (riskCanvas) {
            new Chart(riskCanvas, {
                type: 'doughnut',
                data: {
                    labels: ['At Risk', 'On Track / Safe'],
                    datasets: [{
                        data: [DASHBOARD_DATA.atRisk, DASHBOARD_DATA.notAtRisk],
                        backgroundColor: ['#ef4444', '#10b981'],
                        hoverBackgroundColor: ['#dc2626', '#059669'],
                        borderWidth: 2,
                        borderColor: '#ffffff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom' },
                        tooltip: {
                            callbacks: {
                                label: function (ctx) {
                                    const total = DASHBOARD_DATA.atRisk + DASHBOARD_DATA.notAtRisk;
                                    const pct = total > 0 ? ((ctx.parsed / total) * 100).toFixed(1) : 0;
                                    return ` ${ctx.label}: ${ctx.parsed} (${pct}%)`;
                                }
                            }
                        }
                    },
                    cutout: '68%'
                }
            });
        }

        // 2. Grade Progression (G1 vs G2) Bar Chart
        const gradeCanvas = document.getElementById('gradeProgressionChart');
        if (gradeCanvas && DASHBOARD_DATA.students) {
            const sampleStudents = DASHBOARD_DATA.students.slice(0, 7);
            const studentLabels = sampleStudents.map(s => s.name.split(' ')[0]);
            const g1Data = sampleStudents.map(s => s.g1);
            const g2Data = sampleStudents.map(s => s.g2);

            new Chart(gradeCanvas, {
                type: 'bar',
                data: {
                    labels: studentLabels,
                    datasets: [
                        {
                            label: 'Period 1 (G1)',
                            data: g1Data,
                            backgroundColor: 'rgba(59, 130, 246, 0.75)',
                            borderColor: '#2563eb',
                            borderWidth: 1,
                            borderRadius: 4
                        },
                        {
                            label: 'Period 2 (G2)',
                            data: g2Data,
                            backgroundColor: 'rgba(16, 185, 129, 0.75)',
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
                            title: { display: true, text: 'Exam Score (out of 20)' }
                        }
                    },
                    plugins: {
                        legend: { position: 'top' }
                    }
                }
            });
        }

        // 3. K-Means Cluster Distribution Chart
        const clusterCanvas = document.getElementById('clusterDistributionChart');
        if (clusterCanvas && DASHBOARD_DATA.clusterDist) {
            const clusterLabels = Object.keys(DASHBOARD_DATA.clusterDist);
            const clusterCounts = Object.values(DASHBOARD_DATA.clusterDist);

            new Chart(clusterCanvas, {
                type: 'polarArea',
                data: {
                    labels: clusterLabels,
                    datasets: [{
                        data: clusterCounts,
                        backgroundColor: [
                            'rgba(245, 158, 11, 0.75)',
                            'rgba(16, 185, 129, 0.75)',
                            'rgba(239, 68, 68, 0.75)'
                        ],
                        borderColor: '#ffffff',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 11 } } }
                    }
                }
            });
        }

        // 4. Absences vs Risk Probability Chart
        const absencesCanvas = document.getElementById('absencesRiskChart');
        if (absencesCanvas && DASHBOARD_DATA.students) {
            const sortedByAbsences = [...DASHBOARD_DATA.students].sort((a, b) => a.absences - b.absences);
            const labels = sortedByAbsences.map(s => s.name.split(' ')[0]);
            const absenceVals = sortedByAbsences.map(s => s.absences);
            const riskVals = sortedByAbsences.map(s => Math.round(s.risk_probability * 100));

            new Chart(absencesCanvas, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Risk Probability (%)',
                            data: riskVals,
                            borderColor: '#ef4444',
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            fill: true,
                            tension: 0.3,
                            yAxisID: 'y'
                        },
                        {
                            label: 'Absences (Days)',
                            data: absenceVals,
                            borderColor: '#3b82f6',
                            backgroundColor: 'transparent',
                            borderDash: [5, 5],
                            tension: 0.2,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: { display: true, text: 'Risk %' },
                            max: 100
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: { display: true, text: 'Absences' },
                            grid: { drawOnChartArea: false }
                        }
                    },
                    plugins: {
                        legend: { position: 'top' }
                    }
                }
            });
        }
    }

    // -------------------------------------------------------------
    // REPORTS & ANALYTICS CHARTS
    // -------------------------------------------------------------
    if (typeof REPORT_DATA !== 'undefined') {

        // 1. Feature Importance Horizontal Bar Chart
        const featCanvas = document.getElementById('featureImportanceChart');
        if (featCanvas && REPORT_DATA.rfFeatures) {
            const labels = REPORT_DATA.rfFeatures.map(f => f.feature);
            const values = REPORT_DATA.rfFeatures.map(f => f.importance);

            new Chart(featCanvas, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Gini Importance Score',
                        data: values,
                        backgroundColor: 'rgba(37, 99, 235, 0.8)',
                        hoverBackgroundColor: '#1d4ed8',
                        borderRadius: 6
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            beginAtZero: true,
                            title: { display: true, text: 'Relative Feature Contribution (Total = 1.0)' }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }

        // 2. K-Means Elbow Curve Chart
        const elbowCanvas = document.getElementById('elbowChart');
        if (elbowCanvas && REPORT_DATA.elbowK && REPORT_DATA.elbowInertias) {
            new Chart(elbowCanvas, {
                type: 'line',
                data: {
                    labels: REPORT_DATA.elbowK.map(k => `k=${k}`),
                    datasets: [{
                        label: 'Inertia (Sum of Squared Distances)',
                        data: REPORT_DATA.elbowInertias,
                        borderColor: '#2563eb',
                        backgroundColor: 'rgba(37, 99, 235, 0.1)',
                        pointBackgroundColor: REPORT_DATA.elbowK.map(k => k === 3 ? '#ef4444' : '#2563eb'),
                        pointRadius: REPORT_DATA.elbowK.map(k => k === 3 ? 8 : 4),
                        pointHoverRadius: 9,
                        fill: true,
                        tension: 0.25
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            title: { display: true, text: 'Inertia (Cluster SSE)' }
                        },
                        x: {
                            title: { display: true, text: 'Number of Clusters (k)' }
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                afterLabel: function (ctx) {
                                    if (ctx.dataIndex === 2) {
                                        return '★ Selected Optimal Elbow Point (k=3)';
                                    }
                                    return '';
                                }
                            }
                        }
                    }
                }
            });
        }
    }

});
