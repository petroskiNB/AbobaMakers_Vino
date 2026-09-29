"""Plot the recorded 100-image HTTPS timing run without excluding failures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/latency100_20260929'


def main():
    summary = json.loads((OUT / 'run/summary.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (OUT / 'run/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    times = [r['latency_ms'] / 1000 for r in rows]
    fig, (ax, hist) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Время ответа сайта: 100 разных изображений из data/foto', fontsize=16)
    ax.plot(range(1, len(rows)+1), times, color='#781b43', linewidth=1)
    failed = [(i+1, t) for i, (r, t) in enumerate(zip(rows, times)) if not r['ok']]
    if failed:
        ax.scatter(*zip(*failed), marker='x', color='red', label='Ошибка запроса')
    ax.axhline(3, color='#d16b16', linestyle='--', label='Целевой SLA: < 3 с')
    ax.set(xlabel='Номер запроса', ylabel='Время, с', ylim=(0, None))
    ax.legend()
    hist.hist(times, bins=15, color='#781b43', edgecolor='white')
    hist.axvline(3, color='#d16b16', linestyle='--')
    hist.set(xlabel='Время, с', ylabel='Число запросов')
    m = summary['latency_ms']
    fig.text(.05, .05, f"Среднее {m['mean']/1000:.2f} с · p50 {m['p50']/1000:.2f} с · p95 {m['p95_nearest_rank']/1000:.2f} с · "
             f"Успешно быстрее 3 с: {summary['successful_under_3s']}/100", fontsize=11)
    fig.text(.05, .01, 'HTTPS с компьютера команды; последовательные запросы, без повторов и исключения первого. Не оценка точности.', fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, .94))
    for suffix in ('png', 'svg'):
        fig.savefig(OUT / f'latency100.{suffix}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
