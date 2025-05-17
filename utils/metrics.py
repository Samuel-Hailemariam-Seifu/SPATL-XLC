# metrics_logger.py
import time
import torch
import numpy as np
import pandas as pd
import os
from sklearn.metrics import silhouette_score
import csv

class MetricsTracker:
    def __init__(self, num_rounds, target_acc=70.0, log_dir="./logs"):
        self.num_rounds = num_rounds
        self.target_acc = target_acc
        self.log_dir = log_dir
        self.rounds_completed = 0

        # Core metrics
        self.global_acc = []
        self.global_top5 = []
        self.round_times = []
        self.rounds_to_target = None
        self.best_acc = 0.0

        # Efficiency
        self.pruning_ratios = []
        self.flop_reductions = []
        self.shap_times = []

        # Robustness & fairness
        self.client_accuracies = []
        self.client_drifts = []
        self.shap_entropies = []
        self.cluster_features = []
        self.silhouette_scores = []

        self.start_time = time.time()

    def start_round(self):
        self.round_start = time.time()

    def end_round(self):
        self.round_times.append(time.time() - self.round_start)

    def update_global_metrics(self, acc, top5):
        self.global_acc.append(acc)
        self.global_top5.append(top5)
        if acc > self.best_acc:
            self.best_acc = acc
        if acc >= self.target_acc and self.rounds_to_target is None:
            self.rounds_to_target = len(self.global_acc)

    def update_pruning_metrics(self, pruning, flops):
        self.pruning_ratios.append(pruning)
        self.flop_reductions.append(flops)

    def update_shap_time(self, seconds):
        self.shap_times.append(seconds)

    def update_client_metrics(self, client_accs, client_drift):
        self.client_accuracies.append(client_accs)
        self.client_drifts.append(client_drift)
        

    def update_shap_entropy(self, entropy):
        self.mean_entropy = np.mean(self.shap_entropies) if self.shap_entropies else None
        self.shap_entropies.append(entropy)

    def update_cluster_features(self, features, labels):
        self.cluster_features.append(features)
        if len(set(labels)) < 2 or len(features) < 3:
            self.silhouette_scores.append(-1)
            return
        try:
            score = silhouette_score(np.vstack(features), labels)
            self.silhouette_scores.append(score)
        except Exception:
            self.silhouette_scores.append(-1)


    def finalize(self, final_accuracy):
        self.total_time = time.time() - self.start_time
        self.avg_time_per_round = np.mean(self.round_times)
        self.std_acc = np.std([np.mean(accs) for accs in self.client_accuracies])
        self.avg_pruning = np.mean(self.pruning_ratios)
        self.avg_flops = np.mean(self.flop_reductions)
        self.avg_shap_time = np.mean(self.shap_times) if self.shap_times else 0
        self.mean_entropy = np.mean(self.shap_entropies) if self.shap_entropies else None
        self.mean_drift = np.mean(self.client_drifts) if self.client_drifts else None
        self.avg_silhouette = np.mean(self.silhouette_scores) if self.silhouette_scores else None

        self.summary = {
            "Final Accuracy": final_accuracy,
            "Best Accuracy": self.best_acc,
            "Rounds to Target":  self.rounds_completed,
            "Top-5 Accuracy": self.global_top5[-1] if self.global_top5 else 0,
            "Accuracy Std": self.std_acc,
            "Total Time": self.total_time,
            "Time/Round": self.avg_time_per_round,
            "FLOPs Reduction": self.avg_flops,
            "Param Reduction": self.avg_pruning,
            "SHAP Time": self.avg_shap_time,
            "SHAP Entropy": self.mean_entropy,
            "Client Drift": self.mean_drift,
            "Silhouette Score": self.avg_silhouette,
        }

        self.save_summary()

    def save_summary(self):
        os.makedirs(self.log_dir, exist_ok=True)
    
        # Save global summary
        df = pd.DataFrame([self.summary])
        df.to_csv(os.path.join(self.log_dir, "metrics_summary.csv"), index=False)
        print("[MetricsTracker] Summary saved to", self.log_dir)
    
        # Save per-client metrics
        client_metrics_path = os.path.join(self.log_dir, "client_metrics.csv")
        with open(client_metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "client_id", "accuracy", "drift"])
    
            for round_idx, (acc_list, drift_list) in enumerate(zip(self.client_accuracies, self.client_drifts)):
                for client_id, (acc, drift) in enumerate(zip(acc_list, drift_list)):
                    writer.writerow([round_idx, client_id, acc, drift])
    
        print("[MetricsTracker] Client metrics saved to", client_metrics_path)
    
        

    def get_summary(self):
        return self.summary
