from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent

# API response summary
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#0d1b2a')
ax.set_facecolor('#0d1b2a')
ax.axis('off')
ax.text(0.05, 0.75, 'API Response', color='white', fontsize=22, weight='bold')
ax.text(0.05, 0.52, '{\n  "prediction": 0,\n  "risk": "Low risk",\n  "confidence": 0.7585\n}', color='#7fd1ff', fontsize=16, family='monospace')
fig.savefig(base / 'api_response.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# CI summary
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#f5f5f5')
ax.set_facecolor('#f5f5f5')
ax.axis('off')
ax.text(0.05, 0.8, 'CI/CD Workflow', fontsize=22, weight='bold', color='#1f2937')
ax.text(0.12, 0.62, '1. Install dependencies', fontsize=14, color='#1f2937')
ax.text(0.12, 0.52, '2. Run unit tests', fontsize=14, color='#1f2937')
ax.text(0.12, 0.42, '3. Train model', fontsize=14, color='#1f2937')
ax.text(0.12, 0.32, '4. Upload artifacts', fontsize=14, color='#1f2937')
fig.savefig(base / 'ci_workflow.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# Deployment summary
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#e8f0fe')
ax.set_facecolor('#e8f0fe')
ax.axis('off')
ax.text(0.05, 0.8, 'Deployment Summary', fontsize=22, weight='bold', color='#0b3d91')
ax.text(0.08, 0.6, 'Docker image built', fontsize=16, color='#0b3d91')
ax.text(0.08, 0.48, 'Kubernetes manifest created', fontsize=16, color='#0b3d91')
ax.text(0.08, 0.36, 'Model served on /predict', fontsize=16, color='#0b3d91')
fig.savefig(base / 'deployment_summary.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# Monitoring summary
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#edf7ed')
ax.set_facecolor('#edf7ed')
ax.axis('off')
ax.text(0.05, 0.8, 'Monitoring & Metrics', fontsize=22, weight='bold', color='#1f6f3a')
ax.text(0.08, 0.6, 'Prometheus metrics exposed at /metrics', fontsize=15, color='#1f6f3a')
ax.text(0.08, 0.48, 'Request counts and latency tracked', fontsize=15, color='#1f6f3a')
ax.text(0.08, 0.36, 'Logging enabled for API calls', fontsize=15, color='#1f6f3a')
fig.savefig(base / 'monitoring_summary.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# Build a complete slideshow for the video with visible content on every slide
slides = [
    ('pipeline_intro.png', 'Heart Disease MLOps Pipeline\n\nData -> EDA -> Train -> Track -> Deploy', '#111827', '#ffffff'),
    ('api_response.png', 'Prediction API\n\nPOST /predict\nreturns label + confidence', '#0d1b2a', '#7fd1ff'),
    ('class_distribution.png', 'EDA Output\n\nClass balance: 164 healthy, 139 diseased', '#1d3557', '#f1faee'),
    ('ci_workflow.png', 'CI/CD Workflow\n\nInstall -> Test -> Train -> Artifacts', '#f5f5f5', '#1f2937'),
    ('deployment_summary.png', 'Deployment\n\nDockerized API + Kubernetes manifest', '#e8f0fe', '#0b3d91'),
    ('monitoring_summary.png', 'Monitoring\n\nPrometheus metrics + request logging', '#edf7ed', '#1f6f3a'),
]

for name, title, color, text_color in slides:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(color)
    ax.set_facecolor(color)
    ax.axis('off')
    ax.text(0.5, 0.72, title.split('\n')[0], ha='center', va='center', fontsize=22, weight='bold', color=text_color)
    body = '\n'.join(title.split('\n')[1:]) if '\n' in title else ''
    if body:
        ax.text(0.5, 0.42, body, ha='center', va='center', fontsize=16, color=text_color, linespacing=1.6)
    fig.savefig(base / name, dpi=200, bbox_inches='tight')
    plt.close(fig)

print('Screenshots generated in', base)
