"""
ICIP Report Figure Generator
Generates publication-quality figures and LaTeX tables for ICIP paper
"""
import json
import csv
import os
import numpy as np
from pathlib import Path

# Try importing optional visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class ICIPReportGenerator:
    """Generate figures and tables for ICIP paper"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / 'results'
        self.output_dir = self.project_root / 'report_figures'
        self.output_dir.mkdir(exist_ok=True)
        
    def load_training_history(self) -> dict:
        """Load training history from checkpoint"""
        history_path = self.project_root / 'checkpoints' / 'training_history.json'
        with open(history_path, 'r') as f:
            return json.load(f)
    
    def load_comparison_results(self) -> list:
        """Load comparison results CSV"""
        csv_path = self.results_dir / 'final_comparison_100_epochs' / 'comparison_results.csv'
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    def generate_latex_table_metrics(self) -> str:
        """Generate LaTeX table for quality metrics"""
        rows = self.load_comparison_results()
        
        # Group by preset
        presets = {}
        for row in rows:
            preset = row['preset']
            if preset not in presets:
                presets[preset] = []
            presets[preset].append({k: float(v) if k not in ['image', 'preset'] else v 
                                   for k, v in row.items()})
        
        metrics = ['entropy', 'contrast', 'sharpness', 'colorfulness', 'brightness']
        preset_order = ['baseline', 'minimal', 'balanced', 'aggressive']
        
        latex = r"""
\begin{table}[t]
\centering
\caption{Average Quality Metrics Across 100 Test Images}
\label{tab:metrics}
\begin{tabular}{lccccc}
\toprule
Preset & Entropy & Contrast & Sharpness & Colorfulness & Brightness \\
\midrule
"""
        
        for preset in preset_order:
            data = presets[preset]
            row = f"{preset.capitalize()} "
            for metric in metrics:
                values = [d[metric] for d in data]
                mean = np.mean(values)
                std = np.std(values)
                row += f"& {mean:.2f}$\\pm${std:.2f} "
            latex += row + r"\\" + "\n"
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        return latex
    
    def generate_latex_table_improvement(self) -> str:
        """Generate LaTeX table for improvement percentages"""
        rows = self.load_comparison_results()
        
        # Group by preset
        presets = {}
        for row in rows:
            preset = row['preset']
            if preset not in presets:
                presets[preset] = []
            presets[preset].append({k: float(v) if k not in ['image', 'preset'] else v 
                                   for k, v in row.items()})
        
        metrics = ['entropy', 'contrast', 'sharpness', 'colorfulness']
        
        # Calculate baseline means
        baseline_data = presets['baseline']
        baseline_means = {m: np.mean([d[m] for d in baseline_data]) for m in metrics}
        
        latex = r"""
\begin{table}[t]
\centering
\caption{Improvement Over Baseline (\%)}
\label{tab:improvement}
\begin{tabular}{lcccc}
\toprule
Preset & Entropy & Contrast & Sharpness & Colorfulness \\
\midrule
"""
        
        for preset in ['minimal', 'balanced', 'aggressive']:
            data = presets[preset]
            row = f"{preset.capitalize()} "
            for metric in metrics:
                mean = np.mean([d[metric] for d in data])
                improvement = ((mean - baseline_means[metric]) / baseline_means[metric]) * 100
                sign = '+' if improvement > 0 else ''
                row += f"& {sign}{improvement:.1f}\\% "
            latex += row + r"\\" + "\n"
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        return latex
    
    def generate_latex_table_training(self) -> str:
        """Generate LaTeX table for training configuration"""
        latex = r"""
\begin{table}[t]
\centering
\caption{Training Configuration}
\label{tab:training}
\begin{tabular}{ll}
\toprule
Parameter & Value \\
\midrule
Dataset & LOL Dataset \\
Training Samples & 689 pairs \\
Validation Samples & 100 pairs \\
Batch Size & 8 \\
Patch Size & 48$\times$48 \\
Optimizer & Adam \\
Learning Rate & 0.001 $\rightarrow$ 0.0001 \\
Epochs & 100 \\
Parameters & 555,205 \\
Training Time & $\sim$18 minutes \\
\bottomrule
\end{tabular}
\end{table}
"""
        return latex
    
    def generate_training_curve(self):
        """Generate training curve figure"""
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available, skipping training curve generation")
            return None
            
        history = self.load_training_history()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='LR Decay')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([1, 100])
        
        # Learning rate
        lr = [0.001] * 50 + [0.0001] * 49 + [0.00001]
        ax2.semilogy(epochs, lr, 'g-', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Learning Rate (log scale)', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([1, 100])
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'training_curves.pdf'
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.savefig(self.output_dir / 'training_curves.png', format='png', bbox_inches='tight', dpi=300)
        plt.close()
        
        return output_path
    
    def generate_metrics_bar_chart(self):
        """Generate bar chart comparing presets"""
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available, skipping bar chart generation")
            return None
            
        rows = self.load_comparison_results()
        
        # Group by preset
        presets = {}
        for row in rows:
            preset = row['preset']
            if preset not in presets:
                presets[preset] = []
            presets[preset].append({k: float(v) if k not in ['image', 'preset'] else v 
                                   for k, v in row.items()})
        
        preset_order = ['baseline', 'minimal', 'balanced', 'aggressive']
        metrics = ['entropy', 'contrast', 'colorfulness']
        
        # Calculate means
        data = {m: [] for m in metrics}
        for preset in preset_order:
            for m in metrics:
                values = [d[m] for d in presets[preset]]
                data[m].append(np.mean(values))
        
        # Normalize for visualization
        normalized = {m: np.array(data[m]) / max(data[m]) * 100 for m in metrics}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(preset_order))
        width = 0.25
        
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            bars = ax.bar(x + i * width, normalized[metric], width, label=metric.capitalize(), color=color)
        
        ax.set_xlabel('Enhancement Preset', fontsize=12)
        ax.set_ylabel('Normalized Score (%)', fontsize=12)
        ax.set_title('Quality Metrics Comparison by Preset', fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels([p.capitalize() for p in preset_order])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 120])
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'metrics_comparison.pdf'
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.savefig(self.output_dir / 'metrics_comparison.png', format='png', bbox_inches='tight', dpi=300)
        plt.close()
        
        return output_path
    
    def generate_all(self):
        """Generate all report assets"""
        print("=" * 60)
        print("ICIP Report Generator")
        print("=" * 60)
        
        # Generate LaTeX tables
        print("\n1. Generating LaTeX Tables...")
        
        tables_path = self.output_dir / 'latex_tables.tex'
        with open(tables_path, 'w') as f:
            f.write("% ICIP Paper LaTeX Tables\n")
            f.write("% Auto-generated\n\n")
            f.write(self.generate_latex_table_training())
            f.write("\n")
            f.write(self.generate_latex_table_metrics())
            f.write("\n")
            f.write(self.generate_latex_table_improvement())
        print(f"   -> Saved to: {tables_path}")
        
        # Generate figures
        print("\n2. Generating Figures...")
        
        if HAS_MATPLOTLIB:
            training_fig = self.generate_training_curve()
            if training_fig:
                print(f"   -> Training curves: {training_fig}")
            
            metrics_fig = self.generate_metrics_bar_chart()
            if metrics_fig:
                print(f"   -> Metrics comparison: {metrics_fig}")
        else:
            print("   -> Skipped (matplotlib not available)")
        
        # Summary statistics
        print("\n3. Summary Statistics")
        print("-" * 40)
        
        history = self.load_training_history()
        print(f"   Training Epochs: {len(history['train_loss'])}")
        print(f"   Best Val Loss: {min(history['val_loss']):.4f}")
        print(f"   Final Train Loss: {history['train_loss'][-1]:.4f}")
        
        rows = self.load_comparison_results()
        print(f"   Test Images: {len(set(r['image'] for r in rows))}")
        print(f"   Presets Tested: {len(set(r['preset'] for r in rows))}")
        
        print("\n" + "=" * 60)
        print("Report assets generated successfully!")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)


def main():
    import sys
    
    # Determine project root
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = os.path.dirname(os.path.abspath(__file__))
    
    generator = ICIPReportGenerator(project_root)
    generator.generate_all()


if __name__ == '__main__':
    main()
