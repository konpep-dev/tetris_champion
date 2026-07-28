import json
import os

HEADER = """# Tetris AI Training History

| gen | best_score | avg_score | best_steps | avg_steps | best_lines | avg_lines | best_fit | avg_fit |
|-----|-----------|-----------|-----------|----------|-----------|----------|---------|--------|
"""

class Tracker:
    def __init__(self, path="history.md", json_path="history.json"):
        self.path = path
        self.json_path = json_path
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write(HEADER)

    def log(self, entry):
        row = (
            f"{entry['gen']} | "
            f"{entry['best_score']} | "
            f"{entry['avg_score']:.1f} | "
            f"{entry['best_steps']} | "
            f"{entry['avg_steps']:.1f} | "
            f"{entry['best_lines']} | "
            f"{entry['avg_lines']:.1f} | "
            f"{entry['best_fit']:.0f} | "
            f"{entry['avg_fit']:.0f}\n"
        )
        with open(self.path, 'a') as f:
            f.write(row)
        history = self.load_json()
        history.append(entry)
        with open(self.json_path, 'w') as f:
            json.dump(history, f, indent=2)

    def load_json(self):
        if os.path.exists(self.json_path):
            with open(self.json_path) as f:
                return json.load(f)
        return []

    def close(self):
        pass
