"""
Reptile Algorithm Implementation
Context7 Enterprise Pattern: First-Order Meta-Learning for Crypto Trading

Реализация алгоритма Reptile - упрощенной версии MAML без вторых производных.
Особенно эффективен для быстрой адаптации к новым криптовалютным активам.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm
from collections import OrderedDict
import copy

from ..utils.gradient_utils import GradientManager
from ..utils.meta_utils import MetaLearningMetrics


@dataclass
class ReptileConfig:
    """Конфигурация для Reptile алгоритма"""
    
    # Основные параметры
    inner_lr: float = 0.01  # Скорость обучения на задаче
    meta_lr: float = 0.001  # Скорость мета-обучения
    num_inner_steps: int = 5  # Количество шагов на задаче
    
    # Параметры задач
    num_support: int = 5  # Размер support set
    num_query: int = 15  # Размер query set
    
    # Оптимизация
    meta_batch_size: int = 32  # Количество задач в мета-batch
    gradient_clip: Optional[float] = 1.0  # Обрезка градиентов
    weight_decay: float = 0.0001  # L2 регуляризация
    
    # Мониторинг
    log_interval: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Reptile:
    """
    Reptile Meta-Learning Algorithm
    
    Context7 Pattern: Simplified Meta-Learning System
    - First-order optimization only
    - Memory efficient
    - Fast convergence
    - Production scalable
    
    Reptile использует простое правило обновления:
    θ' = θ + ε * (φ - θ)
    где φ - параметры после адаптации на задаче
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ReptileConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация Reptile
        
        Args:
            model: Базовая модель для мета-обучения
            config: Конфигурация Reptile
            logger: Логгер для мониторинга
        """
        self.model = model.to(config.device)
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Мета-оптимизатор не нужен, используем прямое обновление
        self.gradient_manager = GradientManager()
        self.metrics = MetaLearningMetrics()
        
        # Состояние
        self.global_step = 0
        self.best_meta_loss = float('inf')
        
        self.logger.info(f"Reptile initialized with config: {config}")
    
    def inner_adaptation(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor
    ) -> Tuple[OrderedDict, List[float]]:
        """
        Адаптация модели к конкретной задаче
        
        Args:
            support_data: Данные для обучения на задаче
            support_labels: Метки для обучения
            
        Returns:
            Tuple из адаптированных параметров и losses
        """
        # Создаем копию модели для адаптации
        adapted_model = copy.deepcopy(self.model)
        
        # Оптимизатор для адаптации
        inner_optimizer = optim.SGD(
            adapted_model.parameters(),
            lr=self.config.inner_lr
        )
        
        adaptation_losses = []
        
        for step in range(self.config.num_inner_steps):
            inner_optimizer.zero_grad()
            
            # Forward pass
            predictions = adapted_model(support_data)
            loss = nn.functional.mse_loss(predictions, support_labels)
            adaptation_losses.append(loss.item())
            
            # Backward pass
            loss.backward()
            inner_optimizer.step()
        
        # Возвращаем адаптированные параметры
        adapted_params = OrderedDict(adapted_model.named_parameters())
        return adapted_params, adaptation_losses
    
    def meta_train_step(
        self,
        task_batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Один шаг мета-обучения Reptile
        
        Args:
            task_batch: Batch задач для мета-обучения
            
        Returns:
            Словарь с метриками
        """
        # Сохраняем исходные параметры
        original_params = OrderedDict()
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
        
        all_adaptation_losses = []
        query_losses = []
        query_accuracies = []
        adapted_params_list = []
        
        # Обрабатываем каждую задачу в batch
        for task in task_batch:
            support_data = task['support_data'].to(self.config.device)
            support_labels = task['support_labels'].to(self.config.device)
            query_data = task['query_data'].to(self.config.device)
            query_labels = task['query_labels'].to(self.config.device)
            
            # Адаптация к задаче
            adapted_params, adaptation_losses = self.inner_adaptation(
                support_data, support_labels
            )
            adapted_params_list.append(adapted_params)
            all_adaptation_losses.extend(adaptation_losses)
            
            # Оценка на query set
            with torch.no_grad():
                # Временно применяем адаптированные параметры
                self._apply_params(adapted_params)
                
                query_predictions = self.model(query_data)
                query_loss = nn.functional.mse_loss(
                    query_predictions, query_labels
                ).item()
                query_losses.append(query_loss)
                
                query_accuracy = self._compute_accuracy(
                    query_predictions, query_labels
                )
                query_accuracies.append(query_accuracy)
                
                # Восстанавливаем исходные параметры
                self._apply_params(original_params)
        
        # Reptile мета-обновление
        self._reptile_meta_update(adapted_params_list, original_params)
        
        # Метрики
        metrics = {
            'adaptation_loss': np.mean(all_adaptation_losses),
            'query_loss': np.mean(query_losses),
            'query_accuracy': np.mean(query_accuracies),
            'gradient_norm': self.gradient_manager.compute_gradient_norm(
                self.model.parameters()
            )
        }
        
        self.global_step += 1
        return metrics
    
    def _reptile_meta_update(
        self,
        adapted_params_list: List[OrderedDict],
        original_params: OrderedDict
    ) -> None:
        """
        Основное обновление Reptile
        
        Args:
            adapted_params_list: Список адаптированных параметров
            original_params: Исходные параметры модели
        """
        # Вычисляем среднее направление обновления
        meta_gradients = OrderedDict()
        
        for name in original_params.keys():
            # Средняя разность между адаптированными и исходными параметрами
            param_diffs = []
            for adapted_params in adapted_params_list:
                diff = adapted_params[name].data - original_params[name]
                param_diffs.append(diff)
            
            # Усредняем по всем задачам
            meta_gradients[name] = torch.stack(param_diffs).mean(dim=0)
        
        # Применяем мета-обновление
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in meta_gradients:
                    # Reptile update: θ = θ + α * (φ_avg - θ)
                    param.data.add_(
                        meta_gradients[name], alpha=self.config.meta_lr
                    )
                    
                    # Weight decay
                    if self.config.weight_decay > 0:
                        param.data.mul_(1 - self.config.weight_decay)
        
        # Gradient clipping на мета-градиенты
        if self.config.gradient_clip:
            total_norm = 0
            for grad in meta_gradients.values():
                total_norm += grad.norm().item() ** 2
            total_norm = total_norm ** 0.5
            
            if total_norm > self.config.gradient_clip:
                clip_coef = self.config.gradient_clip / (total_norm + 1e-6)
                for name, param in self.model.named_parameters():
                    if name in meta_gradients:
                        param.data.add_(
                            meta_gradients[name] * (clip_coef - 1),
                            alpha=self.config.meta_lr
                        )
    
    def _apply_params(self, params: OrderedDict) -> None:
        """Применяет параметры к модели"""
        for name, param in self.model.named_parameters():
            if name in params:
                param.data = params[name].data
    
    def meta_validate(
        self,
        validation_tasks: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Валидация мета-модели
        
        Args:
            validation_tasks: Задачи для валидации
            
        Returns:
            Словарь с метриками валидации
        """
        # Сохраняем текущие параметры
        original_params = OrderedDict()
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
        
        all_metrics = []
        
        try:
            for task in validation_tasks:
                support_data = task['support_data'].to(self.config.device)
                support_labels = task['support_labels'].to(self.config.device)
                query_data = task['query_data'].to(self.config.device)
                query_labels = task['query_labels'].to(self.config.device)
                
                # Адаптация к задаче
                adapted_params, adaptation_losses = self.inner_adaptation(
                    support_data, support_labels
                )
                
                # Применяем адаптированные параметры
                self._apply_params(adapted_params)
                
                # Оценка на query set
                with torch.no_grad():
                    query_predictions = self.model(query_data)
                    query_loss = nn.functional.mse_loss(
                        query_predictions, query_labels
                    ).item()
                    
                    query_accuracy = self._compute_accuracy(
                        query_predictions, query_labels
                    )
                
                all_metrics.append({
                    'query_loss': query_loss,
                    'query_accuracy': query_accuracy,
                    'adaptation_loss': np.mean(adaptation_losses)
                })
                
                # Восстанавливаем исходные параметры для следующей задачи
                self._apply_params(original_params)
        
        finally:
            # Гарантированно восстанавливаем параметры
            self._apply_params(original_params)
        
        # Агрегируем метрики
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[f'val_{key}'] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def few_shot_adapt(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        num_adaptation_steps: Optional[int] = None,
        return_copy: bool = True
    ) -> nn.Module:
        """
        Быстрая адаптация к новой задаче
        
        Args:
            support_data: Данные для адаптации
            support_labels: Метки для адаптации
            num_adaptation_steps: Количество шагов адаптации
            return_copy: Возвращать копию или изменить исходную модель
            
        Returns:
            Адаптированная модель
        """
        if num_adaptation_steps is None:
            num_adaptation_steps = self.config.num_inner_steps
        
        if return_copy:
            adapted_model = copy.deepcopy(self.model)
        else:
            adapted_model = self.model
        
        # Оптимизатор для адаптации
        adaptation_optimizer = optim.SGD(
            adapted_model.parameters(),
            lr=self.config.inner_lr
        )
        
        # Адаптация
        adapted_model.train()
        for step in range(num_adaptation_steps):
            adaptation_optimizer.zero_grad()
            
            predictions = adapted_model(support_data)
            loss = nn.functional.mse_loss(predictions, support_labels)
            
            loss.backward()
            adaptation_optimizer.step()
            
            if step % 5 == 0:
                self.logger.debug(f"Adaptation step {step}, loss: {loss.item():.4f}")
        
        return adapted_model
    
    def _compute_accuracy(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.1
    ) -> float:
        """Вычисляет accuracy для регрессии"""
        with torch.no_grad():
            errors = torch.abs(predictions - labels)
            correct = (errors < threshold).float()
            return correct.mean().item()
    
    def save_checkpoint(self, filepath: str) -> None:
        """Сохранение checkpoint модели"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'best_meta_loss': self.best_meta_loss
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Reptile checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Загрузка checkpoint модели"""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_meta_loss = checkpoint['best_meta_loss']
        
        self.logger.info(f"Reptile checkpoint loaded from {filepath}")


class ReptileTrainer:
    """
    Trainer для Reptile с Context7 patterns
    
    Features:
    - Efficient memory usage
    - Fast training cycles
    - Robust checkpoint management
    - Comprehensive metrics
    """
    
    def __init__(
        self,
        reptile: Reptile,
        train_loader: Any,
        val_loader: Any,
        config: ReptileConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.reptile = reptile
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.metrics_history = []
        self.best_val_accuracy = 0.0
    
    def train(
        self,
        num_iterations: int,
        save_dir: str = "./checkpoints",
        validation_interval: int = 100
    ) -> Dict[str, List[float]]:
        """
        Основной цикл обучения Reptile
        
        Args:
            num_iterations: Количество итераций обучения
            save_dir: Директория для checkpoint'ов
            validation_interval: Интервал валидации
            
        Returns:
            История метрик обучения
        """
        iteration = 0
        
        for epoch in range(num_iterations // len(self.train_loader) + 1):
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                if iteration >= num_iterations:
                    break
                
                # Training step
                train_metrics = self.reptile.meta_train_step(batch)
                
                # Validation
                if iteration % validation_interval == 0:
                    val_metrics = self.reptile.meta_validate(self.val_loader)
                    
                    # Combine metrics
                    combined_metrics = {**train_metrics, **val_metrics}
                    self.metrics_history.append(combined_metrics)
                    
                    # Logging
                    if iteration % self.config.log_interval == 0:
                        self._log_metrics(iteration, combined_metrics)
                    
                    # Checkpoint saving
                    current_val_accuracy = val_metrics.get('val_query_accuracy', 0)
                    if current_val_accuracy > self.best_val_accuracy:
                        self.best_val_accuracy = current_val_accuracy
                        self.reptile.save_checkpoint(f"{save_dir}/best_reptile_model.pt")
                
                iteration += 1
        
        return self._compile_metrics_history()
    
    def _log_metrics(self, iteration: int, metrics: Dict[str, float]) -> None:
        """Логирование метрик"""
        self.logger.info(f"Iteration {iteration}:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
    
    def _compile_metrics_history(self) -> Dict[str, List[float]]:
        """Компиляция истории метрик"""
        if not self.metrics_history:
            return {}
        
        compiled = {}
        for key in self.metrics_history[0].keys():
            compiled[key] = [m[key] for m in self.metrics_history]
        return compiled