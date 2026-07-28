import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#00d2ff', '#ffc832', '#ff4444', '#44ff88', '#ff66cc']

def load_history(path="history.json"):
    if not os.path.exists(path):
        print(f"No history found at {path}. Run the simulation first.")
        return []
    with open(path) as f:
        return json.load(f)

def plot_scores(history, out="chart_scores.png"):
    if not history:
        return
    gens = [h['gen'] for h in history]
    best = [h['best_score'] for h in history]
    avg = [h['avg_score'] for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, best, color=COLORS[0], linewidth=2, label='Best Score')
    ax.plot(gens, avg, color=COLORS[1], linewidth=1.5, linestyle='--', label='Avg Score')
    ax.fill_between(gens, avg, best, alpha=0.08, color=COLORS[0])
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Tetris AI — Score Progression', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

def plot_steps(history, out="chart_steps.png"):
    if not history:
        return
    gens = [h['gen'] for h in history]
    best = [h['best_steps'] for h in history]
    avg = [h['avg_steps'] for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, best, color=COLORS[2], linewidth=2, label='Best Steps')
    ax.plot(gens, avg, color=COLORS[3], linewidth=1.5, linestyle='--', label='Avg Steps')
    ax.fill_between(gens, avg, best, alpha=0.08, color=COLORS[2])
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Steps (pieces placed)', fontsize=12)
    ax.set_title('Tetris AI — Steps Progression', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

def plot_lines(history, out="chart_lines.png"):
    if not history:
        return
    gens = [h['gen'] for h in history]
    best = [h['best_lines'] for h in history]
    avg = [h['avg_lines'] for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gens, best, color=COLORS[0], linewidth=2, label='Best Lines')
    ax.plot(gens, avg, color=COLORS[4], linewidth=1.5, linestyle='--', label='Avg Lines')
    ax.fill_between(gens, avg, best, alpha=0.08, color=COLORS[0])
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Lines Cleared', fontsize=12)
    ax.set_title('Tetris AI — Lines Cleared Progression', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")

def plot_all(history):
    plot_scores(history)
    plot_steps(history)
    plot_lines(history)

if __name__ == "__main__":
    h = load_history()
    if h:
        plot_all(h)
    else:
        print("No data. Run python main.py first to generate history.")
