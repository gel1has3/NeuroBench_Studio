/**
 * EEG Foundation Model Dashboard - Interactive Charts
 * Plotly.js visualizations for dimensionality, CKA, and manifold data
 */

(function() {
    'use strict';

    // ==============================================================
    // Color Palette
    // ==============================================================
    const COLORS = {
        models: ['#0dcaf0', '#0d6efd', '#6f42c1', '#198754', '#fd7e14', '#dc3545', '#20c997', '#e83e8c'],
        diseases: ['#0dcaf0', '#ffc107', '#198754', '#dc3545', '#6f42c1'],
        background: getComputedStyle(document.documentElement).getPropertyValue('--card-bg').trim() || '#1e1e32',
        text: getComputedStyle(document.documentElement).getPropertyValue('--text-muted').trim() || '#8888aa',
        grid: getComputedStyle(document.documentElement).getPropertyValue('--card-border').trim() || '#2a2a45',
    };

    // ==============================================================
    // Plotly Layout Templates
    // ==============================================================
    
    function getLayout(title, height = 400, extraProps = {}) {
        const isDark = document.documentElement.getAttribute('data-bs-theme') === 'dark';
        
        return {
            title: {
                text: title,
                font: { size: 13, color: isDark ? '#e0e0ff' : '#212529' },
                x: 0.02,
                y: 0.95,
            },
            height: height,
            margin: { l: 60, r: 20, t: 40, b: 60 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: isDark ? '#8888aa' : '#6c757d' },
            xaxis: {
                gridcolor: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.08)',
                zerolinecolor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                tickfont: { size: 11 },
            },
            yaxis: {
                gridcolor: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.08)',
                zerolinecolor: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                tickfont: { size: 11 },
            },
            hoverlabel: {
                bgcolor: isDark ? '#1e1e32' : '#ffffff',
                font: { color: isDark ? '#e0e0ff' : '#212529' },
                bordercolor: isDark ? '#2a2a45' : '#dee2e6',
            },
            legend: {
                font: { size: 11, color: isDark ? '#8888aa' : '#6c757d' },
                orientation: 'h',
                y: -0.2,
                x: 0.5,
                xanchor: 'center',
            },
            ...extraProps,
        };
    }

    // ==============================================================
    // Overview: Dimensionality Bar Chart
    // ==============================================================
    
    function renderOverviewDimChart(containerId) {
        const el = document.getElementById(containerId);
        if (!el || !window.DIM_DATA || window.DIM_DATA.length === 0) return;

        const data = window.DIM_DATA;
        const models = [...new Set(data.map(d => d.model))];
        const diseases = [...new Set(data.map(d => d.disease))];
        
        const traces = models.map((model, idx) => {
            const modelData = data.filter(d => d.model === model);
            return {
                x: modelData.map(d => d.disease),
                y: modelData.map(d => d.participation_ratio || 0),
                name: model,
                type: 'bar',
                marker: { color: COLORS.models[idx % COLORS.models.length], opacity: 0.85 },
                error_y: {
                    type: 'data',
                    array: modelData.map(d => (d.participation_ratio || 0) * 0.1),
                    visible: true,
                    color: COLORS.models[idx % COLORS.models.length],
                },
                hovertemplate: '<b>%{x}</b><br>Model: ' + model + '<br>PR: %{y:.2f}<extra></extra>',
            };
        });

        const layout = getLayout('Intrinsic Dimensionality by Model and Disease', 350, {
            barmode: 'group',
            yaxis: { title: 'Participation Ratio', gridcolor: COLORS.grid },
            xaxis: { automargin: true },
            showlegend: true,
        });

        Plotly.newPlot(el, traces, layout, {
            responsive: true,
            displayModeBar: false,
        });
    }

    // ==============================================================
    // Overview: Manifold Structure Comparison (Grouped Bar)
    // ==============================================================
    
    function renderOverviewManifoldChart(containerId) {
        const el = document.getElementById(containerId);
        if (!el || !window.MANIFOLD_DATA) return;

        const data = window.MANIFOLD_DATA;
        const models = Object.keys(data);
        if (models.length === 0) return;

        const metrics = ['cluster_purity', 'disease_mixing_score', 'mean_knn_overlap'];
        const metricLabels = ['Cluster Purity', 'Disease Mixing', 'KNN Overlap'];

        const traces = metrics.map((metric, idx) => ({
            x: models,
            y: models.map(m => {
                const val = data[m][metric];
                return val !== null && val !== undefined ? val : 0;
            }),
            name: metricLabels[idx],
            type: 'bar',
            marker: { color: COLORS.models[idx % COLORS.models.length], opacity: 0.85 },
            hovertemplate: '<b>%{x}</b><br>' + metricLabels[idx] + ': %{y:.3f}<extra></extra>',
        }));

        const layout = getLayout('Manifold Structure Comparison', 350, {
            barmode: 'group',
            yaxis: { title: 'Score', range: [0, 1], gridcolor: COLORS.grid },
            showlegend: true,
        });

        Plotly.newPlot(el, traces, layout, {
            responsive: true,
            displayModeBar: false,
        });
    }

    // ==============================================================
    // Dimensionality: Bar Chart (Detailed)
    // ==============================================================
    
    function renderDimBarChart(containerId) {
        const el = document.getElementById(containerId);
        if (!el || !window.DIM_DATA || window.DIM_DATA.length === 0) return;

        const data = window.DIM_DATA;
        const models = [...new Set(data.map(d => d.model))];
        const diseases = [...new Set(data.map(d => d.disease))];

        // Create grouped bar chart
        const traces = [];
        models.forEach((model, idx) => {
            const modelData = data.filter(d => d.model === model);
            traces.push({
                x: modelData.map(d => d.disease),
                y: modelData.map(d => d.participation_ratio || 0),
                name: model,
                type: 'bar',
                marker: { color: COLORS.models[idx % COLORS.models.length], opacity: 0.85 },
                hovertemplate: '<b>%{x}</b><br>Model: ' + model + '<br>PR: %{y:.2f}<br>EffRank: %{customdata:.2f}<extra></extra>',
                customdata: modelData.map(d => d.effective_rank || 0),
            });
        });

        const layout = getLayout('Participation Ratio by Model and Disease', 450, {
            barmode: 'group',
            yaxis: { title: 'Participation Ratio', gridcolor: COLORS.grid },
            xaxis: { automargin: true, tickangle: -15 },
            showlegend: true,
        });

        Plotly.newPlot(el, traces, layout, {
            responsive: true,
            displayModeBar: false,
        });
    }

    // ==============================================================
    // Dimensionality: Heatmap
    // ==============================================================
    
    function renderDimHeatmap(containerId) {
        const el = document.getElementById(containerId);
        if (!el || !window.DIM_DATA || window.DIM_DATA.length === 0) return;

        const data = window.DIM_DATA;
        const models = [...new Set(data.map(d => d.model))];
        const diseases = [...new Set(data.map(d => d.disease))];

        const z = models.map(model => 
            diseases.map(disease => {
                const entry = data.find(d => d.model === model && d.disease === disease);
                return entry ? (entry.participation_ratio || 0) : 0;
            })
        );

        const trace = {
            z: z,
            x: diseases,
            y: models,
            type: 'heatmap',
            colorscale: [
                [0, '#0a0a2e'],
                [0.25, '#1a1a6e'],
                [0.5, '#0d6efd'],
                [0.75, '#0dcaf0'],
                [1, '#f0f0ff'],
            ],
            hovertemplate: '<b>%{y}</b> × <b>%{x}</b><br>PR: %{z:.2f}<extra></extra>',
            colorbar: {
                title: 'PR',
                titleside: 'right',
                thickness: 15,
                len: 0.8,
            },
        };

        const layout = getLayout('Dimensionality Heatmap', 400, {
            xaxis: { automargin: true },
            yaxis: { automargin: true },
            margin: { l: 100, r: 30, t: 30, b: 80 },
        });

        Plotly.newPlot(el, [trace], layout, {
            responsive: true,
            displayModeBar: false,
        });
    }

    // ==============================================================
    // CKA: Bar Chart
    // ==============================================================
    
    function renderCkaBarChart(containerId) {
        const el = document.getElementById(containerId);
        if (!el || !window.CKA_DATA || window.CKA_DATA.length === 0) return;

        // Sort by CKA value
        const sorted = [...window.CKA_DATA].sort((a, b) => a.cka - b.cka);

        const trace = {
            x: sorted.map(d => d.cka),
            y: sorted.map(d => d.pair),
            type: 'bar',
            orientation: 'h',
            marker: {
                color: sorted.map(d => {
                    const norm = Math.min(Math.max(d.cka * 100, 0), 1);
                    return `rgba(13, 202, 240, ${Math.max(norm, 0.2)})`;
                }),
                line: { color: '#0dcaf0', width: 1 },
            },
            hovertemplate: '<b>%{y}</b><br>CKA: %{x:.6f}<br>Samples: %{customdata}<extra></extra>',
            customdata: sorted.map(d => d.n_samples),
        };

        const layout = getLayout('Cross-Model Similarity (CKA)', 450, {
            xaxis: { title: 'CKA Similarity (0 = different, 1 = identical)', gridcolor: COLORS.grid },
            yaxis: { automargin: true, tickfont: { size: 10 } },
            margin: { l: 180, r: 30, t: 40, b: 60 },
            showlegend: false,
        });

        Plotly.newPlot(el, [trace], layout, {
            responsive: true,
            displayModeBar: false,
        });
    }

    // ==============================================================
    // CKA: Heatmap
    // ==============================================================
    
    function renderCkaHeatmap(containerId) {
        const el = document.getElementById(containerId);
        if (!el || !window.CKA_DATA || window.CKA_DATA.length === 0) return;

        // Build matrix from pairwise data
        const models = window.MODEL_NAMES || [];
        if (models.length === 0) return;

        const n = models.length;
        const z = Array(n).fill().map(() => Array(n).fill(0));

        window.CKA_DATA.forEach(item => {
            const [a, b] = item.pair_key.split('_vs_');
            const i = models.indexOf(a);
            const j = models.indexOf(b);
            if (i >= 0 && j >= 0) {
                z[i][j] = item.cka;
                z[j][i] = item.cka;
            }
        });

        // Set diagonal to 1
        for (let i = 0; i < n; i++) z[i][i] = 1;

        const trace = {
            z: z,
            x: models,
            y: models,
            type: 'heatmap',
            colorscale: [
                [0, '#0a0a2e'],
                [0.25, '#1a1a6e'],
                [0.5, '#0d6efd'],
                [0.75, '#0dcaf0'],
                [1, '#f0f0ff'],
            ],
            hovertemplate: '<b>%{y}</b> × <b>%{x}</b><br>CKA: %{z:.6f}<extra></extra>',
            zmin: 0,
            zmax: 1,
            colorbar: {
                title: 'CKA',
                titleside: 'right',
                thickness: 15,
                len: 0.8,
            },
        };

        const layout = getLayout('CKA Similarity Matrix', 400, {
            xaxis: { automargin: true, tickangle: -45 },
            yaxis: { automargin: true },
            margin: { l: 100, r: 30, t: 30, b: 100 },
        });

        Plotly.newPlot(el, [trace], layout, {
            responsive: true,
            displayModeBar: false,
        });
    }

    // ==============================================================
    // Manifold: Radar Chart
    // ==============================================================
    
    function renderManifoldRadarChart(containerId) {
        const el = document.getElementById(containerId);
        if (!el || !window.MANIFOLD_DATA) return;

        const data = window.MANIFOLD_DATA;
        const models = Object.keys(data);
        if (models.length === 0) return;

        const traces = models.map((model, idx) => {
            const ms = data[model];
            const r = [
                ms.disease_mixing_score !== null && ms.disease_mixing_score !== undefined ? ms.disease_mixing_score : 0,
                ms.cluster_purity || 0,
                ms.mean_knn_overlap !== null && ms.mean_knn_overlap !== undefined ? ms.mean_knn_overlap : 0,
            ];
            return {
                type: 'scatterpolar',
                r: [...r, r[0]],
                theta: ['Disease Mixing', 'Cluster Purity', 'KNN Overlap', 'Disease Mixing'],
                fill: 'toself',
                name: model,
                marker: { color: COLORS.models[idx % COLORS.models.length] },
                line: { color: COLORS.models[idx % COLORS.models.length], width: 2 },
                hovertemplate: '<b>' + model + '</b><br>%{theta}: %{r:.3f}<extra></extra>',
            };
        });

        const layout = getLayout('Manifold Structure Comparison', 450, {
            polar: {
                radialaxis: {
                    visible: true,
                    range: [0, 1],
                    gridcolor: COLORS.grid,
                    tickfont: { size: 10 },
                },
                angularaxis: {
                    gridcolor: COLORS.grid,
                    tickfont: { size: 11 },
                },
                bgcolor: 'rgba(0,0,0,0)',
            },
            showlegend: true,
            margin: { l: 80, r: 80, t: 40, b: 80 },
        });

        Plotly.newPlot(el, traces, layout, {
            responsive: true,
            displayModeBar: false,
        });
    }

    // ==============================================================
    // KNN Overlap Matrix Heatmap
    // ==============================================================
    
    function updateKNNHeatmap() {
        const select = document.getElementById('knnModelSelect');
        const el = document.getElementById('knnHeatmap');
        if (!select || !el || !window.MANIFOLD_DATA) return;

        const model = select.value;
        const ms = window.MANIFOLD_DATA[model];
        if (!ms || !ms.knn_overlap_matrix || !ms.diseases) return;

        const matrix = ms.knn_overlap_matrix;
        const diseases = ms.diseases;

        // Set diagonal to 1 for display
        const z = matrix.map((row, i) => 
            row.map((val, j) => (i === j ? 1 : val))
        );

        const trace = {
            z: z,
            x: diseases,
            y: diseases,
            type: 'heatmap',
            colorscale: [
                [0, '#0a0a2e'],
                [0.5, '#0d6efd'],
                [0.8, '#0dcaf0'],
                [1, '#f0f0ff'],
            ],
            hovertemplate: '<b>%{y}</b> × <b>%{x}</b><br>Overlap: %{z:.3f}<extra></extra>',
            zmin: 0.7,
            zmax: 1.0,
            colorbar: {
                title: 'Overlap',
                titleside: 'right',
                thickness: 12,
                len: 0.7,
            },
        };

        const layout = getLayout('KNN Overlap: ' + model, 350, {
            xaxis: { automargin: true, tickangle: -30 },
            yaxis: { automargin: true },
            margin: { l: 80, r: 30, t: 30, b: 80 },
        });

        Plotly.newPlot(el, [trace], layout, {
            responsive: true,
            displayModeBar: false,
        });
    }

    // ==============================================================
    // Re-render charts on theme change
    // ==============================================================
    
    function reRenderAllCharts() {
        const charts = [
            'overviewDimChart',
            'overviewManifoldChart',
            'dimBarChart',
            'dimHeatmap',
            'ckaBarChart',
            'ckaHeatmap',
            'manifoldRadarChart',
            'knnHeatmap',
        ];
        
        charts.forEach(id => {
            const el = document.getElementById(id);
            if (el && el.data) {
                // Charts will re-render on next interaction
                // This forces a refresh
                Plotly.Plots.resize(el);
            }
        });
    }

    // ==============================================================
    // Initialization
    // ==============================================================
    
    document.addEventListener('DOMContentLoaded', function() {
        // Wait a brief moment for all data to be available
        setTimeout(() => {
            // Overview tab charts
            renderOverviewDimChart('overviewDimChart');
            renderOverviewManifoldChart('overviewManifoldChart');
            
            // Dimensionality tab charts
            renderDimBarChart('dimBarChart');
            renderDimHeatmap('dimHeatmap');
            
            // CKA tab charts
            renderCkaBarChart('ckaBarChart');
            renderCkaHeatmap('ckaHeatmap');
            
            // Manifold tab charts
            renderManifoldRadarChart('manifoldRadarChart');
            updateKNNHeatmap();
            
            // Update overview counts
            updateOverviewStats();
        }, 100);
        
        // Re-render charts when theme changes
        const themeBtn = document.getElementById('themeToggle');
        if (themeBtn) {
            themeBtn.addEventListener('click', function() {
                setTimeout(reRenderAllCharts, 300);
            });
        }
        
        // Re-render on tab change
        const tabs = document.querySelectorAll('[data-bs-toggle="tab"]');
        tabs.forEach(tab => {
            tab.addEventListener('shown.bs.tab', function() {
                setTimeout(() => {
                    Plotly.Plots.resize(document.getElementById('overviewDimChart'));
                    Plotly.Plots.resize(document.getElementById('overviewManifoldChart'));
                    Plotly.Plots.resize(document.getElementById('dimBarChart'));
                    Plotly.Plots.resize(document.getElementById('dimHeatmap'));
                    Plotly.Plots.resize(document.getElementById('ckaBarChart'));
                    Plotly.Plots.resize(document.getElementById('ckaHeatmap'));
                    Plotly.Plots.resize(document.getElementById('manifoldRadarChart'));
                    Plotly.Plots.resize(document.getElementById('knnHeatmap'));
                }, 200);
            });
        });
    });

    // ==============================================================
    // Helper: Update Overview Statistics
    // ==============================================================
    
    function updateOverviewStats() {
        // Count unique diseases from DIM_DATA
        const dimEl = document.getElementById('overviewDiseases');
        if (dimEl && window.DIM_DATA) {
            const diseases = new Set(window.DIM_DATA.map(d => d.disease));
            dimEl.textContent = diseases.size;
        }
        
        // Calculate average dimensionality
        const avgDimEl = document.getElementById('overviewAvgDim');
        if (avgDimEl && window.DIM_DATA) {
            const prs = window.DIM_DATA.map(d => d.participation_ratio || 0);
            if (prs.length > 0) {
                const avg = prs.reduce((a, b) => a + b, 0) / prs.length;
                avgDimEl.textContent = avg.toFixed(2);
            }
        }
    }

    // Expose for inline usage
    window.updateKNNHeatmap = updateKNNHeatmap;

})();