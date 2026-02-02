"""
Meta-Learning Utilities
Comprehensive Meta-Learning Support

Набор утилит для мета-обучения: метрики, визуализация, анализ данных,
и вспомогательные функции для криптовалютного трейдинга.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
import time
import pickle
import json
from pathlib import Path
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class MetaLearningMetrics:
    """
    Система метрик для мета-обучения
    
    Comprehensive Metrics System
    - Task-specific metrics
    - Adaptation performance tracking
    - Cross-task generalization analysis
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # История метрик
        self.metrics_history = defaultdict(list)
        self.task_metrics = defaultdict(dict)
        
    def compute_adaptation_metrics(
        self,
        initial_performance: float,
        final_performance: float,
        num_adaptation_steps: int,
        adaptation_time: float
    ) -> Dict[str, float]:
        """
        Вычисляет метрики адаптации
        
        Args:
            initial_performance: Начальная производительность
            final_performance: Финальная производительность
            num_adaptation_steps: Количество шагов адаптации
            adaptation_time: Время адаптации в секундах
            
        Returns:
            Словарь с метриками адаптации
        """
        improvement = final_performance - initial_performance
        improvement_rate = improvement / num_adaptation_steps if num_adaptation_steps > 0 else 0
        
        metrics = {
            'adaptation_improvement': improvement,
            'adaptation_rate': improvement_rate,
            'adaptation_efficiency': improvement / adaptation_time if adaptation_time > 0 else 0,
            'relative_improvement': improvement / abs(initial_performance) if initial_performance != 0 else 0,
            'convergence_speed': 1.0 / num_adaptation_steps if improvement > 0 else 0
        }
        
        return metrics
    
    def compute_few_shot_metrics(
        self,
        support_performance: float,
        query_performance: float,
        num_shots: int,
        num_ways: int
    ) -> Dict[str, float]:
        """
        Вычисляет метрики few-shot обучения
        
        Args:
            support_performance: Производительность на support set
            query_performance: Производительность на query set
            num_shots: Количество примеров на класс
            num_ways: Количество классов
            
        Returns:
            Словарь с метриками few-shot обучения
        """
        generalization_gap = support_performance - query_performance
        sample_efficiency = query_performance / (num_shots * num_ways)
        
        metrics = {
            'generalization_gap': generalization_gap,
            'sample_efficiency': sample_efficiency,
            'query_performance': query_performance,
            'support_performance': support_performance,
            'overfitting_indicator': max(0, generalization_gap),
            'data_efficiency_score': query_performance * np.log(1 + 1/(num_shots * num_ways))
        }
        
        return metrics
    
    def compute_cross_task_metrics(
        self,
        task_performances: Dict[str, float],
        task_similarities: Optional[Dict[Tuple[str, str], float]] = None
    ) -> Dict[str, float]:
        """
        Вычисляет метрики кросс-задачного обобщения
        
        Args:
            task_performances: Производительность по задачам
            task_similarities: Схожесть между задачами
            
        Returns:
            Словарь с метриками кросс-задачного обобщения
        """
        performances = list(task_performances.values())
        
        metrics = {
            'mean_performance': np.mean(performances),
            'std_performance': np.std(performances),
            'min_performance': np.min(performances),
            'max_performance': np.max(performances),
            'performance_range': np.max(performances) - np.min(performances),
            'coefficient_of_variation': np.std(performances) / np.mean(performances) if np.mean(performances) != 0 else 0
        }
        
        # Если есть информация о схожести задач
        if task_similarities:
            # Корреляция между схожестью задач и различием в производительности
            task_names = list(task_performances.keys())
            performance_diffs = []
            similarities = []
            
            for i, task1 in enumerate(task_names):
                for j, task2 in enumerate(task_names[i+1:], i+1):
                    if (task1, task2) in task_similarities:
                        perf_diff = abs(task_performances[task1] - task_performances[task2])
                        similarity = task_similarities[(task1, task2)]
                        
                        performance_diffs.append(perf_diff)
                        similarities.append(similarity)
            
            if performance_diffs and similarities:
                correlation = np.corrcoef(performance_diffs, similarities)[0, 1]
                metrics['similarity_performance_correlation'] = correlation if not np.isnan(correlation) else 0
        
        return metrics
    
    def compute_meta_learning_efficiency(
        self,
        meta_training_time: float,
        meta_training_episodes: int,
        adaptation_times: List[float],
        final_performances: List[float]
    ) -> Dict[str, float]:
        """
        Вычисляет эффективность мета-обучения
        
        Args:
            meta_training_time: Время мета-обучения
            meta_training_episodes: Количество эпизодов мета-обучения
            adaptation_times: Времена адаптации для новых задач
            final_performances: Финальные производительности
            
        Returns:
            Словарь с метриками эффективности
        """
        avg_adaptation_time = np.mean(adaptation_times) if adaptation_times else 0
        avg_performance = np.mean(final_performances) if final_performances else 0
        
        metrics = {
            'meta_training_efficiency': avg_performance / meta_training_time if meta_training_time > 0 else 0,
            'adaptation_efficiency': avg_performance / avg_adaptation_time if avg_adaptation_time > 0 else 0,
            'total_efficiency': avg_performance / (meta_training_time + avg_adaptation_time) if (meta_training_time + avg_adaptation_time) > 0 else 0,
            'time_to_performance_ratio': meta_training_time / avg_performance if avg_performance > 0 else 0,
            'episodes_efficiency': avg_performance / meta_training_episodes if meta_training_episodes > 0 else 0
        }
        
        return metrics
    
    def track_metric(self, metric_name: str, value: float, task_id: Optional[str] = None) -> None:
        """
        Отслеживает метрику
        
        Args:
            metric_name: Имя метрики
            value: Значение метрики
            task_id: ID задачи (опционально)
        """
        self.metrics_history[metric_name].append({
            'value': value,
            'timestamp': time.time(),
            'task_id': task_id
        })
        
        if task_id:
            self.task_metrics[task_id][metric_name] = value
    
    def get_metric_summary(self, metric_name: str, window: Optional[int] = None) -> Dict[str, float]:
        """
        Возвращает сводку по метрике
        
        Args:
            metric_name: Имя метрики
            window: Размер окна для анализа (последние N значений)
            
        Returns:
            Сводка по метрике
        """
        if metric_name not in self.metrics_history:
            return {}
        
        values = [entry['value'] for entry in self.metrics_history[metric_name]]
        
        if window:
            values = values[-window:]
        
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'count': len(values),
            'trend': self._compute_trend(values) if len(values) > 1 else 0
        }
    
    def _compute_trend(self, values: List[float]) -> float:
        """Вычисляет тренд значений (положительный = улучшение)"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Линейная регрессия для определения тренда
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def export_metrics(self, filepath: str) -> None:
        """Экспортирует метрики в файл"""
        export_data = {
            'metrics_history': dict(self.metrics_history),
            'task_metrics': dict(self.task_metrics),
            'export_timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Metrics exported to {filepath}")
    
    def import_metrics(self, filepath: str) -> None:
        """Импортирует метрики из файла"""
        with open(filepath, 'r') as f:
            import_data = json.load(f)
        
        self.metrics_history.update(import_data.get('metrics_history', {}))
        self.task_metrics.update(import_data.get('task_metrics', {}))
        
        self.logger.info(f"Metrics imported from {filepath}")


class DataAnalyzer:
    """
    Анализатор данных для мета-обучения
    
    Data Analysis Pipeline
    - Task difficulty estimation
    - Data quality assessment
    - Feature importance analysis
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_task_difficulty(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        query_data: torch.Tensor,
        query_labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Анализирует сложность задачи
        
        Args:
            support_data: Support данные
            support_labels: Support метки
            query_data: Query данные
            query_labels: Query метки
            
        Returns:
            Словарь с оценками сложности
        """
        difficulty_metrics = {}
        
        # Конвертируем в numpy для анализа
        support_data_np = support_data.detach().cpu().numpy()
        support_labels_np = support_labels.detach().cpu().numpy()
        query_data_np = query_data.detach().cpu().numpy()
        query_labels_np = query_labels.detach().cpu().numpy()
        
        # 1. Размерность задачи
        difficulty_metrics['feature_dimensionality'] = support_data_np.shape[1] if len(support_data_np.shape) > 1 else 1
        difficulty_metrics['sample_size_ratio'] = len(support_data_np) / len(query_data_np) if len(query_data_np) > 0 else 0
        
        # 2. Сложность распределения данных
        if len(support_data_np.shape) > 1:
            # Оценка сложности через PCA
            try:
                from sklearn.decomposition import PCA
                pca = PCA()
                pca.fit(support_data_np)
                explained_variance_ratio = pca.explained_variance_ratio_
                
                # Интринсивная размерность (90% дисперсии)
                cumulative_variance = np.cumsum(explained_variance_ratio)
                intrinsic_dim = np.argmax(cumulative_variance >= 0.9) + 1
                difficulty_metrics['intrinsic_dimensionality'] = intrinsic_dim
                difficulty_metrics['data_complexity'] = 1.0 - explained_variance_ratio[0]  # Сложность = 1 - доля первой компоненты
            except:
                difficulty_metrics['intrinsic_dimensionality'] = difficulty_metrics['feature_dimensionality']
                difficulty_metrics['data_complexity'] = 0.5
        
        # 3. Сложность классификации/регрессии
        if len(np.unique(support_labels_np)) <= 10:  # Классификация
            # Баланс классов
            unique_labels, counts = np.unique(support_labels_np, return_counts=True)
            class_balance = np.min(counts) / np.max(counts) if len(counts) > 1 else 1.0
            difficulty_metrics['class_imbalance'] = 1.0 - class_balance
            
            # Количество классов
            difficulty_metrics['num_classes'] = len(unique_labels)
            difficulty_metrics['multiclass_complexity'] = len(unique_labels) / 10.0  # Нормализуем
            
        else:  # Регрессия
            # Вариабельность целевой переменной
            target_std = np.std(support_labels_np)
            target_range = np.max(support_labels_np) - np.min(support_labels_np)
            difficulty_metrics['target_variability'] = target_std / (target_range + 1e-8)
            
            # Нелинейность (корреляция с первой главной компонентой)
            if len(support_data_np.shape) > 1 and support_data_np.shape[1] > 1:
                try:
                    first_pc = PCA(n_components=1).fit_transform(support_data_np).flatten()
                    correlation = np.corrcoef(first_pc, support_labels_np)[0, 1]
                    difficulty_metrics['nonlinearity'] = 1.0 - abs(correlation) if not np.isnan(correlation) else 0.5
                except:
                    difficulty_metrics['nonlinearity'] = 0.5
        
        # 4. Схожесть support/query распределений
        support_mean = np.mean(support_data_np, axis=0)
        query_mean = np.mean(query_data_np, axis=0)
        distribution_shift = np.linalg.norm(support_mean - query_mean)
        difficulty_metrics['distribution_shift'] = distribution_shift
        
        # 5. Общая оценка сложности (комбинированная)
        complexity_factors = [
            difficulty_metrics.get('data_complexity', 0.5),
            difficulty_metrics.get('class_imbalance', 0) + difficulty_metrics.get('target_variability', 0),
            difficulty_metrics.get('multiclass_complexity', 0) + difficulty_metrics.get('nonlinearity', 0),
            min(1.0, difficulty_metrics.get('distribution_shift', 0))
        ]
        
        difficulty_metrics['overall_difficulty'] = np.mean(complexity_factors)
        
        return difficulty_metrics
    
    def assess_data_quality(
        self,
        data: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Оценивает качество данных
        
        Args:
            data: Данные для анализа
            labels: Метки
            
        Returns:
            Словарь с оценками качества
        """
        quality_metrics = {}
        
        data_np = data.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # 1. Пропущенные значения
        nan_ratio = np.sum(np.isnan(data_np)) / data_np.size
        quality_metrics['missing_data_ratio'] = nan_ratio
        
        # 2. Выбросы (IQR метод)
        if len(data_np.shape) > 1:
            outlier_ratios = []
            for col in range(data_np.shape[1]):
                column_data = data_np[:, col]
                q1, q3 = np.percentile(column_data, [25, 75])
                iqr = q3 - q1
                outlier_bounds = [q1 - 1.5 * iqr, q3 + 1.5 * iqr]
                outliers = (column_data < outlier_bounds[0]) | (column_data > outlier_bounds[1])
                outlier_ratios.append(np.sum(outliers) / len(column_data))
            
            quality_metrics['outlier_ratio'] = np.mean(outlier_ratios)
        else:
            quality_metrics['outlier_ratio'] = 0.0
        
        # 3. Дубликаты
        if len(data_np.shape) > 1:
            unique_rows = len(np.unique(data_np, axis=0))
            total_rows = len(data_np)
            quality_metrics['duplicate_ratio'] = 1.0 - (unique_rows / total_rows)
        else:
            quality_metrics['duplicate_ratio'] = 0.0
        
        # 4. Консистентность меток
        if len(np.unique(labels_np)) > 1:
            # Для классификации: проверяем сбалансированность
            unique_labels, counts = np.unique(labels_np, return_counts=True)
            balance_score = np.min(counts) / np.max(counts)
            quality_metrics['label_consistency'] = balance_score
        else:
            quality_metrics['label_consistency'] = 1.0
        
        # 5. Шум в данных (оценка через локальную вариабельность)
        if len(data_np) > 5 and len(data_np.shape) > 1:
            try:
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=min(5, len(data_np)))
                nn.fit(data_np)
                distances, _ = nn.kneighbors(data_np)
                avg_distance = np.mean(distances[:, 1:])  # Исключаем расстояние до себя
                noise_estimate = avg_distance / np.std(data_np)
                quality_metrics['noise_level'] = min(1.0, noise_estimate)
            except:
                quality_metrics['noise_level'] = 0.5
        else:
            quality_metrics['noise_level'] = 0.0
        
        # 6. Общая оценка качества
        quality_score = 1.0 - np.mean([
            quality_metrics['missing_data_ratio'],
            quality_metrics['outlier_ratio'],
            quality_metrics['duplicate_ratio'],
            1.0 - quality_metrics['label_consistency'],
            quality_metrics['noise_level']
        ])
        
        quality_metrics['overall_quality'] = max(0.0, quality_score)
        
        return quality_metrics
    
    def compute_feature_importance(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        method: str = "variance"
    ) -> Dict[str, np.ndarray]:
        """
        Вычисляет важность признаков
        
        Args:
            data: Данные
            labels: Метки
            method: Метод вычисления (variance, correlation, mutual_info)
            
        Returns:
            Словарь с важностью признаков
        """
        data_np = data.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        if len(data_np.shape) == 1:
            return {'importance': np.array([1.0]), 'method': method}
        
        importance_scores = np.zeros(data_np.shape[1])
        
        if method == "variance":
            # Важность на основе дисперсии
            importance_scores = np.var(data_np, axis=0)
            importance_scores = importance_scores / (np.sum(importance_scores) + 1e-8)
            
        elif method == "correlation":
            # Важность на основе корреляции с целевой переменной
            for i in range(data_np.shape[1]):
                correlation = np.corrcoef(data_np[:, i], labels_np)[0, 1]
                importance_scores[i] = abs(correlation) if not np.isnan(correlation) else 0
            
            importance_scores = importance_scores / (np.sum(importance_scores) + 1e-8)
            
        elif method == "mutual_info":
            # Важность на основе взаимной информации
            try:
                from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
                
                if len(np.unique(labels_np)) <= 10:  # Классификация
                    importance_scores = mutual_info_classif(data_np, labels_np)
                else:  # Регрессия
                    importance_scores = mutual_info_regression(data_np, labels_np)
                
                importance_scores = importance_scores / (np.sum(importance_scores) + 1e-8)
            except:
                # Fallback к методу корреляции
                return self.compute_feature_importance(data, labels, "correlation")
        
        return {
            'importance': importance_scores,
            'method': method,
            'top_features': np.argsort(importance_scores)[::-1][:10]  # Топ 10 признаков
        }


class Visualizer:
    """
    Система визуализации для мета-обучения
    
    Comprehensive Visualization
    - Training progress visualization
    - Task analysis plots
    - Performance comparison charts
    """
    
    def __init__(
        self,
        save_dir: str = "./plots",
        logger: Optional[logging.Logger] = None
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        
        # Настройка стиля
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_training_progress(
        self,
        metrics_history: Dict[str, List[Dict[str, Any]]],
        save_name: str = "training_progress.png"
    ) -> None:
        """
        Визуализирует прогресс обучения
        
        Args:
            metrics_history: История метрик
            save_name: Имя файла для сохранения
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Meta-Learning Training Progress', fontsize=16)
        
        # Основные метрики
        main_metrics = ['loss', 'accuracy', 'meta_loss', 'adaptation_loss']
        
        for i, metric_name in enumerate(main_metrics):
            ax = axes[i // 2, i % 2]
            
            if metric_name in metrics_history:
                values = [entry['value'] for entry in metrics_history[metric_name]]
                timestamps = [entry['timestamp'] for entry in metrics_history[metric_name]]
                
                # Конвертируем timestamps в относительное время
                if timestamps:
                    start_time = timestamps[0]
                    relative_times = [(t - start_time) / 3600 for t in timestamps]  # В часах
                    
                    ax.plot(relative_times, values, marker='o', markersize=3)
                    ax.set_title(f'{metric_name.replace("_", " ").title()}')
                    ax.set_xlabel('Time (hours)')
                    ax.set_ylabel(metric_name)
                    ax.grid(True, alpha=0.3)
                    
                    # Добавляем тренд
                    if len(values) > 1:
                        z = np.polyfit(relative_times, values, 1)
                        p = np.poly1d(z)
                        ax.plot(relative_times, p(relative_times), "r--", alpha=0.8, label='Trend')
                        ax.legend()
            else:
                ax.text(0.5, 0.5, f'No data for {metric_name}', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes)
                ax.set_title(f'{metric_name.replace("_", " ").title()}')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training progress plot saved to {self.save_dir / save_name}")
    
    def plot_few_shot_performance(
        self,
        performance_data: Dict[str, Dict[str, float]],
        save_name: str = "few_shot_performance.png"
    ) -> None:
        """
        Визуализирует производительность few-shot обучения
        
        Args:
            performance_data: Данные производительности по настройкам
            save_name: Имя файла для сохранения
        """
        # Извлекаем данные для графика
        shots = []
        accuracies = []
        settings = []
        
        for setting_name, metrics in performance_data.items():
            if 'shot' in setting_name and 'accuracy' in metrics:
                # Извлекаем количество shots из имени настройки
                shot_count = int(setting_name.split('shot')[0].split('_')[-1])
                shots.append(shot_count)
                accuracies.append(metrics['accuracy']['mean'])
                settings.append(setting_name)
        
        if not shots:
            self.logger.warning("No few-shot data found for visualization")
            return
        
        # Создаем график
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Few-Shot Learning Performance', fontsize=16)
        
        # График 1: Производительность vs количество shots
        unique_shots = sorted(list(set(shots)))
        mean_accuracies = []
        std_accuracies = []
        
        for shot_count in unique_shots:
            shot_accs = [acc for s, acc in zip(shots, accuracies) if s == shot_count]
            mean_accuracies.append(np.mean(shot_accs))
            std_accuracies.append(np.std(shot_accs) if len(shot_accs) > 1 else 0)
        
        ax1.errorbar(unique_shots, mean_accuracies, yerr=std_accuracies, 
                    marker='o', capsize=5, capthick=2)
        ax1.set_xlabel('Number of Shots')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Performance vs Number of Shots')
        ax1.grid(True, alpha=0.3)
        
        # График 2: Heatmap всех настроек
        if len(performance_data) > 1:
            setting_names = list(performance_data.keys())
            metric_names = ['accuracy', 'precision', 'recall', 'f1']
            
            # Создаем матрицу данных
            data_matrix = []
            available_metrics = []
            
            for metric in metric_names:
                metric_row = []
                has_data = False
                for setting in setting_names:
                    if metric in performance_data[setting]:
                        metric_row.append(performance_data[setting][metric].get('mean', 0))
                        has_data = True
                    else:
                        metric_row.append(0)
                
                if has_data:
                    data_matrix.append(metric_row)
                    available_metrics.append(metric)
            
            if data_matrix:
                sns.heatmap(data_matrix, 
                           xticklabels=[s.replace('_', '\n') for s in setting_names],
                           yticklabels=available_metrics,
                           annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2)
                ax2.set_title('Performance Heatmap')
                ax2.set_xlabel('Settings')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Few-shot performance plot saved to {self.save_dir / save_name}")
    
    def plot_model_comparison(
        self,
        model_results: Dict[str, Dict[str, Any]],
        save_name: str = "model_comparison.png"
    ) -> None:
        """
        Визуализирует сравнение моделей
        
        Args:
            model_results: Результаты различных моделей
            save_name: Имя файла для сохранения
        """
        if len(model_results) < 2:
            self.logger.warning("Need at least 2 models for comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Comparison', fontsize=16)
        
        # Извлекаем данные для сравнения
        model_names = list(model_results.keys())
        
        # График 1: Средняя производительность по моделям
        ax1 = axes[0, 0]
        avg_performances = []
        
        for model_name in model_names:
            # Берем среднюю accuracy по всем настройкам
            accuracies = []
            for setting_results in model_results[model_name]['aggregated_results'].values():
                if 'accuracy' in setting_results:
                    accuracies.append(setting_results['accuracy']['mean'])
            
            avg_performances.append(np.mean(accuracies) if accuracies else 0)
        
        bars = ax1.bar(model_names, avg_performances)
        ax1.set_title('Average Performance')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticklabels(model_names, rotation=45)
        
        # Добавляем значения на столбцы
        for bar, perf in zip(bars, avg_performances):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{perf:.3f}', ha='center', va='bottom')
        
        # График 2: Время обучения (если доступно)
        ax2 = axes[0, 1]
        training_times = []
        
        for model_name in model_names:
            timing_stats = model_results[model_name].get('timing_stats', {})
            if timing_stats:
                avg_time = np.mean(list(timing_stats.values()))
                training_times.append(avg_time)
            else:
                training_times.append(0)
        
        if any(t > 0 for t in training_times):
            ax2.bar(model_names, training_times)
            ax2.set_title('Average Training Time')
            ax2.set_ylabel('Time (seconds)')
            ax2.set_xticklabels(model_names, rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No timing data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes)
            ax2.set_title('Training Time')
        
        # График 3: Scatter plot производительность vs время
        ax3 = axes[1, 0]
        if any(t > 0 for t in training_times):
            scatter = ax3.scatter(training_times, avg_performances, s=100, alpha=0.7)
            
            for i, model_name in enumerate(model_names):
                ax3.annotate(model_name, (training_times[i], avg_performances[i]),
                           xytext=(5, 5), textcoords='offset points')
            
            ax3.set_xlabel('Training Time (seconds)')
            ax3.set_ylabel('Accuracy')
            ax3.set_title('Performance vs Training Time')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No timing data for scatter plot', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes)
            ax3.set_title('Performance vs Training Time')
        
        # График 4: Radar chart сравнения
        ax4 = axes[1, 1]
        self._create_radar_chart(model_results, ax4)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Model comparison plot saved to {self.save_dir / save_name}")
    
    def _create_radar_chart(
        self,
        model_results: Dict[str, Dict[str, Any]],
        ax: plt.Axes
    ) -> None:
        """Создает radar chart для сравнения моделей"""
        try:
            # Извлекаем метрики для сравнения
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            model_names = list(model_results.keys())
            
            # Подготавливаем данные
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Замыкаем круг
            
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
            
            for model_name in model_names:
                values = []
                for metric in metrics:
                    # Берем среднее значение по всем настройкам
                    metric_values = []
                    for setting_results in model_results[model_name]['aggregated_results'].values():
                        if metric in setting_results:
                            metric_values.append(setting_results[metric]['mean'])
                    
                    avg_value = np.mean(metric_values) if metric_values else 0
                    values.append(avg_value)
                
                values += values[:1]  # Замыкаем круг
                
                ax.plot(angles, values, marker='o', label=model_name)
                ax.fill(angles, values, alpha=0.1)
            
            ax.set_ylim(0, 1)
            ax.set_title('Model Performance Radar Chart')
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Radar chart error: {str(e)}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title('Model Performance Radar Chart')


class ModelSerializer:
    """
    Сериализатор для сохранения и загрузки мета-моделей
    
    Model Persistence
    - Model checkpointing
    - Configuration saving
    - Version management
    """
    
    def __init__(
        self,
        save_dir: str = "./checkpoints",
        logger: Optional[logging.Logger] = None
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
    
    def save_model(
        self,
        model: nn.Module,
        model_name: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None
    ) -> str:
        """
        Сохраняет модель с конфигурацией и метаданными
        
        Args:
            model: Модель для сохранения
            model_name: Имя модели
            config: Конфигурация модели
            metadata: Дополнительные метаданные
            version: Версия модели
            
        Returns:
            Путь к сохраненному файлу
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'config': config,
            'metadata': metadata or {},
            'version': version,
            'save_timestamp': time.time(),
            'pytorch_version': torch.__version__
        }
        
        filename = f"{model_name}_v{version}.pt"
        filepath = self.save_dir / filename
        
        torch.save(checkpoint, filepath)
        
        # Сохраняем также конфигурацию отдельно для удобства
        config_filename = f"{model_name}_v{version}_config.json"
        config_filepath = self.save_dir / config_filename
        
        with open(config_filepath, 'w') as f:
            json.dump({
                'config': config,
                'metadata': metadata or {},
                'version': version
            }, f, indent=2, default=str)
        
        self.logger.info(f"Model saved to {filepath}")
        return str(filepath)
    
    def load_model(
        self,
        filepath: str,
        model_class: type,
        device: Optional[torch.device] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Загружает модель из checkpoint
        
        Args:
            filepath: Путь к checkpoint файлу
            model_class: Класс модели
            device: Устройство для загрузки
            
        Returns:
            Tuple из модели и метаданных
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        # Создаем модель
        config = checkpoint['config']
        model = model_class(**config)
        
        # Загружаем веса
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if device:
            model = model.to(device)
        
        metadata = {
            'model_name': checkpoint.get('model_name'),
            'version': checkpoint.get('version'),
            'save_timestamp': checkpoint.get('save_timestamp'),
            'metadata': checkpoint.get('metadata', {})
        }
        
        self.logger.info(f"Model loaded from {filepath}")
        return model, metadata
    
    def list_saved_models(self) -> List[Dict[str, Any]]:
        """Возвращает список сохраненных моделей"""
        models = []
        
        for filepath in self.save_dir.glob("*.pt"):
            try:
                checkpoint = torch.load(filepath, map_location='cpu')
                model_info = {
                    'filepath': str(filepath),
                    'model_name': checkpoint.get('model_name', 'unknown'),
                    'version': checkpoint.get('version', 'unknown'),
                    'save_timestamp': checkpoint.get('save_timestamp'),
                    'file_size': filepath.stat().st_size
                }
                models.append(model_info)
            except Exception as e:
                self.logger.warning(f"Could not load model info from {filepath}: {e}")
        
        # Сортируем по времени сохранения
        models.sort(key=lambda x: x.get('save_timestamp', 0), reverse=True)
        
        return models