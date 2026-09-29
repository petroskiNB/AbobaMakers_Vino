"""Export presentation figures from saved experiments; never merges datasets."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'presentation_charts'
OUT.mkdir(parents=True, exist_ok=True)
metrics = json.loads((ROOT / 'artifacts/experiment/calibration.json').read_text())
bench = json.loads((ROOT / 'artifacts/api_benchmark.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'axes.labelcolor': '#54494d',
                     'text.color': '#30292d', 'axes.edgecolor': '#cbbec3',
                     'figure.facecolor': '#fffaf5', 'axes.facecolor': '#fffaf5',
                     'svg.fonttype': 'none'})
BURGUNDY, GRAY = '#781b43', '#a9a0a5'
rows = []


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=220, facecolor=fig.get_facecolor())
    fig.savefig(OUT / f'{name}.svg', facecolor=fig.get_facecolor())
    plt.close(fig)


def quality(name, key, title, note):
    values = [metrics[group][key] * 100 for group in ['baseline_test', 'selected_test']]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    fig.subplots_adjust(left=.11, right=.95, bottom=.27, top=.80)
    fig.text(.08, .92, title, fontsize=22, weight='bold')
    fig.text(.08, .855, 'Синтетический test · 586 запросов · 293 исходных фото', fontsize=13)
    bars = ax.bar([0, 1], values, color=[GRAY, BURGUNDY], width=.48, zorder=3)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel('%')
    ax.set_xticks([0, 1], ['Базовый энкодер', 'Энкодер + адаптер'])
    ax.grid(axis='y', alpha=.18, zorder=0)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+2, f'{value:.2f}%'.replace('.', ','),
                ha='center', fontsize=20, weight='bold')
    fig.text(.08, .14, 'Прирост: +' + f'{values[1]-values[0]:.2f}'.replace('.', ',') + ' п.п.',
             fontsize=15, color=BURGUNDY, weight='bold')
    fig.text(.08, .07, note, fontsize=10, linespacing=1.5)
    save(fig, name)
    for model, value in zip(['baseline', 'adapter'], values):
        rows.append(['synthetic_test', model, key, value, 'percent', 586])


quality('01_accuracy', 'accuracy_at_1', 'Точность первого кандидата — Accuracy@1',
        'Источник: artifacts/experiment/calibration.json\n'
        'Каталожные фото с синтетическими искажениями; полевая точность не измерена.')
quality('02_macro_f1', 'macro_f1', 'Macro-F1 по классам тестовой выборки',
        'Источник: artifacts/experiment/calibration.json; scripts/calibrate.py\n'
        'Усреднение по 293 целевым классам test, не по всем классам каталога. Не F1 публичного набора.')

times = np.array([m['latency_ms']/1000 for m in bench['measurements']])
mean, median = float(times.mean()), float(np.median(times))
p95 = float(np.sort(times)[int(np.ceil(.95*len(times)))-1])
assert abs(mean*1000-bench['mean_ms']) < .01
assert abs(p95*1000-bench['p95_nearest_rank_ms']) < .01
fig, ax = plt.subplots(figsize=(12, 6.7))
fig.subplots_adjust(left=.09, right=.96, bottom=.29, top=.79)
fig.text(.07, .93, 'Время ответа API на публичных фотографиях', fontsize=22, weight='bold')
fig.text(.07, .865, 'Исторический локальный прогон · прогретый сервис · последовательные запросы', fontsize=12)
ax.bar(np.arange(len(times)), times, color=BURGUNDY, width=.62, zorder=3)
ax.axhline(3, color='#bc652d', linestyle='--', linewidth=2)
ax.text(5.45, 3.07, 'Целевой порог: 3 с', ha='right', color='#975025')
for i, value in enumerate(times):
    ax.text(i, value+.08, f'{value:.3f}'.replace('.', ','), ha='center', weight='bold')
ax.set_ylim(0, 3.55)
ax.set_ylabel('Секунды')
ax.set_xticks(np.arange(len(times)), [f'Фото {i%3+1}\nПовтор {i//3+1}' for i in range(len(times))])
ax.grid(axis='y', alpha=.18, zorder=0)
fig.text(.07, .16, f'Среднее {mean:.3f} с   |   p50 {median:.3f} с   |   p95 {p95:.3f} с   |   < 3 с: {sum(times<3)}/{len(times)}'.replace('.', ','),
         fontsize=14, color=BURGUNDY, weight='bold')
fig.text(.07, .06, 'Источник: artifacts/api_benchmark.json. Три фото, два повтора; p95 — nearest rank.\n'
         'Шесть измерений не подтверждают SLA развёрнутого сервера. Качество ответов здесь не оценивалось.',
         fontsize=10, linespacing=1.5)
save(fig, '03_public_latency')
for metric, value in [('mean', mean), ('p50', median), ('p95_nearest_rank', p95)]:
    rows.append(['public_local_warm_benchmark', 'api', metric, value, 'seconds', len(times)])
with (OUT / 'chart_data.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['dataset', 'model', 'metric', 'value', 'unit', 'queries'])
    writer.writerows(rows)
print(f'Exported three charts (PNG + SVG) and chart_data.csv to {OUT}')
