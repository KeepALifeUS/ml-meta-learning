"""
MAML (Model-Agnostic Meta-Learning) Implementation
Context7 Enterprise Pattern: Scalable Meta-Learning for Crypto Trading

Реализация алгоритма MAML для быстрой адаптации к новым криптовалютным рынкам.
Основана на принципах gradient-based meta-learning с поддержкой высших производных.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm
import higher
from collections import OrderedDict

from ..utils.gradient_utils import GradientManager
from ..utils.meta_utils import MetaLearningMetrics
from ..models.meta_model import MetaModel


@dataclass
class MAMLConfig:
    """Конфигурация для MAML алгоритма"""
    
    # Основные параметры
    inner_lr: float = 0.01  # Скорость обучения на внутреннем цикле
    outer_lr: float = 0.001  # Скорость обучения на внешнем цикле
    num_inner_steps: int = 5  # Количество шагов градиентного спуска на задаче
    
    # Параметры задач
    num_support: int = 5  # Количество примеров в support set
    num_query: int = 15  # Количество примеров в query set
    
    # Регуляризация
    first_order: bool = False  # Использовать ли первый порядок (Reptile-like)
    allow_unused: bool = True  # Разрешить неиспользуемые параметры
    allow_nograd: bool = True  # Разрешить параметры без градиента
    
    # Оптимизация
    grad_clip: Optional[float] = 1.0  # Обрезка градиентов
    weight_decay: float = 0.0001  # L2 регуляризация
    
    # Мониторинг
    log_interval: int = 10  # Интервал логирования
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MAML:
    """
    Model-Agnostic Meta-Learning (MAML) Implementation
    
    Context7 Pattern: Enterprise Meta-Learning System
    - Scalable gradient computation
    - Memory-efficient implementation
    - Production-ready monitoring
    - Crypto market specialization
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: MAMLConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Инициализация MAML
        
        Args:
            model: Базовая модель для мета-обучения
            config: Конфигурация MAML
            logger: Логгер для мониторинга
        """
        self.model = model.to(config.device)
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Оптимизатор для внешнего цикла
        self.meta_optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.outer_lr,
            weight_decay=config.weight_decay
        )
        
        # Утилиты
        self.gradient_manager = GradientManager()
        self.metrics = MetaLearningMetrics()
        
        # Состояние
        self.global_step = 0
        self.best_meta_loss = float('inf')
        
        self.logger.info(f"MAML initialized with config: {config}")
    
    def inner_loop(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        model_state: OrderedDict,
        create_graph: bool = True
    ) -> Tuple[OrderedDict, List[float]]:
        """
        Внутренний цикл обучения на конкретной задаче
        
        Args:
            support_data: Данные для обучения на задаче
            support_labels: Метки для обучения
            model_state: Текущее состояние модели
            create_graph: Создавать ли граф вычислений для второй производной
            
        Returns:
            Tuple из адаптированных параметров и losses
        """
        # Создаем копию модели для внутреннего цикла
        adapted_params = OrderedDict()
        for name, param in model_state.items():
            adapted_params[name] = param.clone()
        
        inner_losses = []
        
        for step in range(self.config.num_inner_steps):
            # Forward pass с текущими параметрами
            predictions = self._forward_with_params(
                support_data, adapted_params
            )
            
            # Вычисляем loss
            loss = nn.functional.mse_loss(predictions, support_labels)
            inner_losses.append(loss.item())
            
            # Вычисляем градиенты
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=create_graph,
                allow_unused=self.config.allow_unused
            )
            
            # Обновляем параметры
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None:
                    adapted_params[name] = param - self.config.inner_lr * grad
        
        return adapted_params, inner_losses
    
    def _forward_with_params(
        self,
        data: torch.Tensor,
        params: OrderedDict
    ) -> torch.Tensor:
        """
        Forward pass с заданными параметрами
        
        Args:
            data: Входные данные
            params: Параметры модели
            
        Returns:
            Предсказания модели
        """
        # Temporary replace model parameters
        original_params = OrderedDict()
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
            if name in params:
                param.data = params[name]
        
        try:
            predictions = self.model(data)
        finally:
            # Restore original parameters
            for name, param in self.model.named_parameters():
                param.data = original_params[name]
        
        return predictions
    
    def meta_train_step(
        self,
        task_batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Один шаг мета-обучения на batch задач
        
        Args:
            task_batch: Batch задач с support/query sets
            
        Returns:
            Словарь с метриками
        """
        self.meta_optimizer.zero_grad()
        
        meta_losses = []
        adaptation_losses = []
        query_accuracies = []
        
        # Получаем текущие параметры модели
        model_state = OrderedDict(self.model.named_parameters())
        
        for task in task_batch:
            support_data = task['support_data'].to(self.config.device)
            support_labels = task['support_labels'].to(self.config.device)
            query_data = task['query_data'].to(self.config.device)
            query_labels = task['query_labels'].to(self.config.device)
            
            # Внутренний цикл - адаптация к задаче
            adapted_params, inner_losses = self.inner_loop(
                support_data,
                support_labels,
                model_state,
                create_graph=not self.config.first_order
            )
            
            # Query loss для внешнего цикла
            query_predictions = self._forward_with_params(
                query_data, adapted_params
            )
            meta_loss = nn.functional.mse_loss(query_predictions, query_labels)
            meta_losses.append(meta_loss)
            
            # Метрики
            adaptation_losses.extend(inner_losses)
            query_accuracy = self._compute_accuracy(
                query_predictions, query_labels
            )
            query_accuracies.append(query_accuracy)
        
        # Агрегируем мета-loss
        total_meta_loss = torch.stack(meta_losses).mean()
        
        # Backward pass
        total_meta_loss.backward()
        
        # Gradient clipping
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
        
        # Обновляем мета-параметры
        self.meta_optimizer.step()
        
        # Собираем метрики
        metrics = {
            'meta_loss': total_meta_loss.item(),
            'adaptation_loss': np.mean(adaptation_losses),
            'query_accuracy': np.mean(query_accuracies),
            'gradient_norm': self.gradient_manager.compute_gradient_norm(
                self.model.parameters()
            )
        }
        
        self.global_step += 1
        return metrics
    
    def meta_validate(
        self,
        validation_tasks: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Валидация мета-модели на validation tasks
        
        Args:
            validation_tasks: Задачи для валидации
            
        Returns:
            Словарь с метриками валидации
        """
        self.model.eval()
        
        all_metrics = []
        
        with torch.no_grad():
            for task in validation_tasks:
                support_data = task['support_data'].to(self.config.device)
                support_labels = task['support_labels'].to(self.config.device)
                query_data = task['query_data'].to(self.config.device)
                query_labels = task['query_labels'].to(self.config.device)
                
                # Адаптация без градиентов для внешнего цикла
                adapted_params, adaptation_losses = self.inner_loop(
                    support_data,
                    support_labels,
                    OrderedDict(self.model.named_parameters()),
                    create_graph=False
                )
                
                # Валидация на query set
                query_predictions = self._forward_with_params(
                    query_data, adapted_params
                )
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
        
        self.model.train()
        
        # Агрегируем метрики
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[f'val_{key}'] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def few_shot_adapt(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        num_adaptation_steps: Optional[int] = None
    ) -> nn.Module:
        """
        Быстрая адаптация к новой задаче (few-shot learning)
        
        Args:
            support_data: Данные для адаптации
            support_labels: Метки для адаптации
            num_adaptation_steps: Количество шагов адаптации
            
        Returns:
            Адаптированная модель
        """
        if num_adaptation_steps is None:
            num_adaptation_steps = self.config.num_inner_steps
        
        # Создаем копию модели для адаптации
        adapted_model = type(self.model)(
            **self.model.config.__dict__ if hasattr(self.model, 'config') else {}
        ).to(self.config.device)
        adapted_model.load_state_dict(self.model.state_dict())
        
        # Optimizer для адаптации
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
        """
        Вычисляет accuracy для регрессии (процент предсказаний в пределах threshold)
        
        Args:
            predictions: Предсказания модели
            labels: Истинные метки
            threshold: Порог для считания предсказания правильным
            
        Returns:
            Accuracy значение
        """
        with torch.no_grad():
            errors = torch.abs(predictions - labels)
            correct = (errors < threshold).float()
            return correct.mean().item()
    
    def save_checkpoint(self, filepath: str) -> None:
        """Сохранение checkpoint модели"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'best_meta_loss': self.best_meta_loss
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Загрузка checkpoint модели"""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_meta_loss = checkpoint['best_meta_loss']
        
        self.logger.info(f"Checkpoint loaded from {filepath}")


class MAMLTrainer:
    """
    Trainer class для MAML с Context7 enterprise patterns
    
    Features:
    - Automated checkpoint management
    - Comprehensive metrics tracking
    - Early stopping
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        maml: MAML,
        train_loader: Any,
        val_loader: Any,
        config: MAMLConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.maml = maml
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Scheduler для learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            maml.meta_optimizer,
            mode='min',
            factor=0.8,
            patience=10,
            verbose=True
        )
        
        self.metrics_history = []
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = "./checkpoints",
        early_stopping_patience: int = 20
    ) -> Dict[str, List[float]]:
        """
        Основной цикл обучения MAML
        
        Args:
            num_epochs: Количество эпох
            save_dir: Директория для сохранения checkpoint'ов
            early_stopping_patience: Терпение для early stopping
            
        Returns:
            История метрик обучения
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self.maml.meta_validate(self.val_loader)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            self.metrics_history.append(epoch_metrics)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['val_query_loss'])
            
            # Logging
            if epoch % self.config.log_interval == 0:
                self._log_metrics(epoch, epoch_metrics)
            
            # Checkpoint saving
            current_val_loss = val_metrics['val_query_loss']
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                self.maml.best_meta_loss = best_val_loss
                self.maml.save_checkpoint(f"{save_dir}/best_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        return self._compile_metrics_history()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Обучение одной эпохи"""
        epoch_metrics = []
        
        for batch in tqdm(self.train_loader, desc="Training"):
            metrics = self.maml.meta_train_step(batch)
            epoch_metrics.append(metrics)
        
        # Агрегируем метрики эпохи
        return {
            key: np.mean([m[key] for m in epoch_metrics])
            for key in epoch_metrics[0].keys()
        }
    
    def _log_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Логирование метрик"""
        self.logger.info(f"Epoch {epoch}:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
    
    def _compile_metrics_history(self) -> Dict[str, List[float]]:
        """Компиляция истории метрик"""
        compiled = {}
        for key in self.metrics_history[0].keys():
            compiled[key] = [m[key] for m in self.metrics_history]
        return compiled