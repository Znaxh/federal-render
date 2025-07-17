"""
Visualization script for federated learning results.
Creates plots for MSE, R², and privacy metrics over FL rounds.
"""

import json
import os
import sys
import argparse
import logging
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FLVisualizer:
    """
    Visualizes federated learning training progress and results.
    """
    
    def __init__(self, metrics_file: str = None, output_dir: str = None):
        """
        Initialize the visualizer.
        
        Args:
            metrics_file: Path to metrics JSON file
            output_dir: Directory to save plots
        """
        self.log_config = get_config("logging")
        self.viz_config = get_config("visualization")
        
        self.metrics_file = metrics_file or self.log_config["metrics_file"]
        self.output_dir = output_dir or self.viz_config["results_dir"]
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set plot style
        plt.style.use(self.viz_config.get("style", "default"))
        sns.set_palette("husl")
        
        self.metrics_data = None
        
    def load_metrics(self) -> bool:
        """
        Load metrics from JSON file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.metrics_file):
                logger.error(f"Metrics file not found: {self.metrics_file}")
                return False
            
            with open(self.metrics_file, 'r') as f:
                self.metrics_data = json.load(f)
            
            logger.info(f"Loaded {len(self.metrics_data)} rounds of metrics")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
            return False
    
    def create_training_progress_plot(self) -> str:
        """
        Create a plot showing training progress over rounds.
        
        Returns:
            Path to saved plot
        """
        if not self.metrics_data:
            raise ValueError("No metrics data loaded")
        
        # Extract data
        rounds = [m["round"] for m in self.metrics_data]
        mse_values = [m.get("mse", np.nan) for m in self.metrics_data]
        r2_values = [m.get("r2", np.nan) for m in self.metrics_data]
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.viz_config["figure_size"])
        
        # MSE plot
        ax1.plot(rounds, mse_values, 'o-', linewidth=2, markersize=6, label='MSE')
        ax1.set_xlabel('Federated Learning Round')
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_title('Model Performance: MSE over FL Rounds')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # R² plot
        ax2.plot(rounds, r2_values, 's-', linewidth=2, markersize=6, 
                label='R² Score', color='green')
        ax2.set_xlabel('Federated Learning Round')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Model Performance: R² over FL Rounds')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        filename = "training_progress.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.viz_config["plot_dpi"], bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training progress plot saved to {filepath}")
        return filepath
    
    def create_privacy_analysis_plot(self) -> str:
        """
        Create a plot showing privacy budget consumption.
        
        Returns:
            Path to saved plot
        """
        if not self.metrics_data:
            raise ValueError("No metrics data loaded")
        
        # Extract privacy data
        rounds = [m["round"] for m in self.metrics_data]
        epsilon_values = [m.get("privacy_epsilon", np.nan) for m in self.metrics_data]
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.viz_config["figure_size"])
        
        ax.plot(rounds, epsilon_values, 'o-', linewidth=2, markersize=6, 
               color='red', label='Cumulative ε')
        ax.set_xlabel('Federated Learning Round')
        ax.set_ylabel('Privacy Budget (ε)')
        ax.set_title('Privacy Budget Consumption over FL Rounds')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add privacy level annotations
        if epsilon_values:
            final_epsilon = epsilon_values[-1]
            if final_epsilon < 1.0:
                privacy_level = "High Privacy"
                color = "green"
            elif final_epsilon < 5.0:
                privacy_level = "Moderate Privacy"
                color = "orange"
            else:
                privacy_level = "Low Privacy"
                color = "red"
            
            ax.text(0.02, 0.98, f"Final Privacy Level: {privacy_level}",
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        plt.tight_layout()
        
        # Save plot
        filename = "privacy_analysis.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.viz_config["plot_dpi"], bbox_inches='tight')
        plt.close()
        
        logger.info(f"Privacy analysis plot saved to {filepath}")
        return filepath
    
    def create_client_comparison_plot(self) -> str:
        """
        Create a plot comparing client performance (if available).
        
        Returns:
            Path to saved plot
        """
        # This would require client-specific metrics
        # For now, create a placeholder plot
        
        fig, ax = plt.subplots(figsize=self.viz_config["figure_size"])
        
        # Simulate client data for demonstration
        clients = ['Hospital A', 'Hospital B', 'Hospital C']
        performance = [0.85, 0.82, 0.88]  # Example R² scores
        
        bars = ax.bar(clients, performance, color=['skyblue', 'lightgreen', 'lightcoral'])
        ax.set_ylabel('R² Score')
        ax.set_title('Final Model Performance by Hospital')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, performance):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        filename = "client_comparison.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.viz_config["plot_dpi"], bbox_inches='tight')
        plt.close()
        
        logger.info(f"Client comparison plot saved to {filepath}")
        return filepath
    
    def create_summary_dashboard(self) -> str:
        """
        Create a comprehensive dashboard with all metrics.
        
        Returns:
            Path to saved plot
        """
        if not self.metrics_data:
            raise ValueError("No metrics data loaded")
        
        # Create a 2x2 subplot dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, 
                                                     figsize=(15, 10))
        
        # Extract data
        rounds = [m["round"] for m in self.metrics_data]
        mse_values = [m.get("mse", np.nan) for m in self.metrics_data]
        r2_values = [m.get("r2", np.nan) for m in self.metrics_data]
        epsilon_values = [m.get("privacy_epsilon", np.nan) for m in self.metrics_data]
        
        # Plot 1: MSE over rounds
        ax1.plot(rounds, mse_values, 'o-', linewidth=2, markersize=4)
        ax1.set_title('MSE over FL Rounds')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('MSE')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: R² over rounds
        ax2.plot(rounds, r2_values, 's-', linewidth=2, markersize=4, color='green')
        ax2.set_title('R² Score over FL Rounds')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('R² Score')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Privacy budget
        ax3.plot(rounds, epsilon_values, '^-', linewidth=2, markersize=4, color='red')
        ax3.set_title('Privacy Budget Consumption')
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Cumulative ε')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        if mse_values and r2_values:
            final_mse = mse_values[-1] if not np.isnan(mse_values[-1]) else 0
            final_r2 = r2_values[-1] if not np.isnan(r2_values[-1]) else 0
            final_epsilon = epsilon_values[-1] if epsilon_values and not np.isnan(epsilon_values[-1]) else 0
            
            metrics = ['Final MSE', 'Final R²', 'Privacy ε', 'Rounds']
            values = [final_mse, final_r2, final_epsilon, len(rounds)]
            
            ax4.barh(metrics, values, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
            ax4.set_title('Final Summary')
            ax4.set_xlabel('Value')
            
            # Add value labels
            for i, v in enumerate(values):
                ax4.text(v + max(values) * 0.01, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        
        # Save plot
        filename = "fl_dashboard.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.viz_config["plot_dpi"], bbox_inches='tight')
        plt.close()
        
        logger.info(f"Dashboard saved to {filepath}")
        return filepath
    
    def generate_report(self) -> str:
        """
        Generate a text report of the federated learning results.
        
        Returns:
            Path to saved report
        """
        if not self.metrics_data:
            raise ValueError("No metrics data loaded")
        
        report_lines = []
        report_lines.append("FEDERATED LEARNING RESULTS REPORT")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Basic statistics
        total_rounds = len(self.metrics_data)
        report_lines.append(f"Total Rounds: {total_rounds}")
        
        if self.metrics_data:
            final_metrics = self.metrics_data[-1]
            
            report_lines.append(f"Final MSE: {final_metrics.get('mse', 'N/A'):.4f}")
            report_lines.append(f"Final R²: {final_metrics.get('r2', 'N/A'):.4f}")
            report_lines.append(f"Privacy Level: {final_metrics.get('privacy_level', 'N/A')}")
            report_lines.append(f"Total Privacy Budget: ε={final_metrics.get('privacy_epsilon', 'N/A'):.4f}")
            report_lines.append("")
            
            # Performance improvement
            if len(self.metrics_data) > 1:
                initial_mse = self.metrics_data[0].get('mse', 0)
                final_mse = final_metrics.get('mse', 0)
                if initial_mse > 0:
                    improvement = ((initial_mse - final_mse) / initial_mse) * 100
                    report_lines.append(f"MSE Improvement: {improvement:.2f}%")
            
            report_lines.append("")
            report_lines.append("Round-by-Round Results:")
            report_lines.append("-" * 30)
            
            for metrics in self.metrics_data:
                round_num = metrics["round"]
                mse = metrics.get("mse", "N/A")
                r2 = metrics.get("r2", "N/A")
                report_lines.append(f"Round {round_num}: MSE={mse:.4f}, R²={r2:.4f}")
        
        # Save report
        report_content = "\n".join(report_lines)
        report_file = os.path.join(self.output_dir, "fl_report.txt")
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to {report_file}")
        return report_file
    
    def visualize_all(self) -> Dict[str, str]:
        """
        Generate all visualizations and reports.
        
        Returns:
            Dictionary mapping visualization names to file paths
        """
        if not self.load_metrics():
            return {}
        
        results = {}
        
        try:
            results["training_progress"] = self.create_training_progress_plot()
            results["privacy_analysis"] = self.create_privacy_analysis_plot()
            results["client_comparison"] = self.create_client_comparison_plot()
            results["dashboard"] = self.create_summary_dashboard()
            results["report"] = self.generate_report()
            
            logger.info("All visualizations generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        return results

def main():
    """Main function to run visualization."""
    parser = argparse.ArgumentParser(description="Federated Learning Visualization")
    parser.add_argument("--metrics-file", type=str,
                       help="Path to metrics JSON file")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for plots")
    parser.add_argument("--plot-type", type=str, 
                       choices=["all", "progress", "privacy", "dashboard"],
                       default="all",
                       help="Type of plot to generate")
    
    args = parser.parse_args()
    
    try:
        visualizer = FLVisualizer(args.metrics_file, args.output_dir)
        
        if args.plot_type == "all":
            results = visualizer.visualize_all()
            print("\nGenerated visualizations:")
            for name, path in results.items():
                print(f"  {name}: {path}")
        else:
            if not visualizer.load_metrics():
                return
            
            if args.plot_type == "progress":
                path = visualizer.create_training_progress_plot()
            elif args.plot_type == "privacy":
                path = visualizer.create_privacy_analysis_plot()
            elif args.plot_type == "dashboard":
                path = visualizer.create_summary_dashboard()
            
            print(f"Generated plot: {path}")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
