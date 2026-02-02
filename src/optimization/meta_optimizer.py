"""
Meta-Optimizer Framework
Advanced Meta-Learning Optimization

Комплексная система мета-оптимизации для быстрой адаптации к новым криптовалютным
рынкам с поддержкой различных алгоритмов и стратегий оптимизации.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
import logging
import numpy as np
from abc import ABC, abstractmethod
import math
from collections import defaultdict, deque
import time

from ..utils.gradient_utils import GradientManager
from ..utils.meta_utils import MetaLearningMetrics


@dataclass
class MetaOptimizerConfig:
    """Конфигурация для мета-оптимизатора"""
    
    # Основные параметры
    meta_lr: float = 0.001  # Мета-скорость обучения
    inner_lr: float = 0.01  # Внутренняя скорость обучения
    num_inner_steps: int = 5  # Количество внутренних шагов
    
    # Оптимизатор
    optimizer_type: str = "adam"  # adam, sgd, rmsprop, adamw
    beta1: float = 0.9  # Для Adam-like оптимизаторов
    beta2: float = 0.999  # Для Adam-like оптимизаторов
    weight_decay: float = 0.0001  # L2 регуляризация
    
    # Градиенты
    grad_clip: Optional[float] = 1.0  # Обрезка градиентов
    grad_accumulation_steps: int = 1  # Аккумуляция градиентов
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # cosine, plateau, exponential, step
    scheduler_patience: int = 10  # Для ReduceLROnPlateau
    scheduler_factor: float = 0.8  # Фактор уменьшения LR
    
    # Adaptive optimization
    use_adaptive_lr: bool = True  # Адаптивная скорость обучения
    lr_adaptation_window: int = 50  # Окно для адаптации
    min_lr: float = 1e-6  # Минимальная скорость обучения
    max_lr: float = 1.0  # Максимальная скорость обучения
    
    # Regularization
    use_weight_decay: bool = True
    use_dropout_schedule: bool = False  # Планирование dropout
    dropout_decay: float = 0.99  # Фактор уменьшения dropout
    
    # Advanced features
    use_gradient_checkpointing: bool = False  # Checkpoint градиентов
    use_mixed_precision: bool = False  # Mixed precision training
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BaseMetaOptimizer(ABC):
    """
    Базовый класс для мета-оптимизаторов
    
    Abstract Meta-Optimizer
    - Pluggable optimization strategies
    - Consistent interface
    - Performance monitoring
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: MetaOptimizerConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Состояние оптимизации
        self.step_count = 0
        self.current_lr = config.meta_lr
        self.best_performance = float('inf')
        
        # Статистики
        self.optimization_history = deque(maxlen=1000)
        self.gradient_stats = defaultdict(list)
        
        # Утилиты
        self.gradient_manager = GradientManager()
        self.metrics = MetaLearningMetrics()
        
    @abstractmethod
    def step(
        self,
        meta_loss: torch.Tensor,
        task_batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Выполняет один шаг мета-оптимизации"""
        pass
    
    @abstractmethod
    def inner_loop_update(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        model_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Выполняет обновление во внутреннем цикле"""
        pass
    
    def zero_grad(self) -> None:
        """Обнуляет градиенты"""
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
    
    def clip_gradients(self) -> float:
        """Обрезает градиенты и возвращает норму"""
        if self.config.grad_clip is None:
            return 0.0
        
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.grad_clip
        )
        return total_norm.item()
    
    def update_learning_rate(self, performance_metric: float) -> None:
        """Обновляет скорость обучения на основе производительности"""
        if not self.config.use_adaptive_lr:
            return
        
        # Простая адаптивная стратегия
        if len(self.optimization_history) >= self.config.lr_adaptation_window:
            recent_performance = [h['performance'] for h in list(self.optimization_history)[-self.config.lr_adaptation_window:]]
            
            # Если производительность не улучшается, уменьшаем LR
            if performance_metric > np.mean(recent_performance):
                self.current_lr = max(
                    self.current_lr * self.config.scheduler_factor,
                    self.config.min_lr
                )
                self.logger.debug(f"Decreased learning rate to {self.current_lr}")
            elif performance_metric < np.min(recent_performance):
                # Если лучший результат, можно увеличить LR
                self.current_lr = min(
                    self.current_lr * 1.05,
                    self.config.max_lr
                )
                self.logger.debug(f"Increased learning rate to {self.current_lr}")
    
    def log_optimization_step(
        self,
        meta_loss: float,
        gradient_norm: float,
        performance_metric: float
    ) -> None:
        """Логирует шаг оптимизации"""
        step_info = {
            'step': self.step_count,
            'meta_loss': meta_loss,
            'gradient_norm': gradient_norm,
            'learning_rate': self.current_lr,
            'performance': performance_metric,
            'timestamp': time.time()
        }
        
        self.optimization_history.append(step_info)
        
        # Обновляем статистики градиентов
        self.gradient_stats['norms'].append(gradient_norm)
        self.gradient_stats['losses'].append(meta_loss)
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Возвращает статистики оптимизации"""
        if not self.optimization_history:
            return {}
        
        recent_history = list(self.optimization_history)[-100:]  # Последние 100 шагов
        
        return {
            'total_steps': self.step_count,
            'current_lr': self.current_lr,
            'best_performance': self.best_performance,
            'recent_avg_loss': np.mean([h['meta_loss'] for h in recent_history]),
            'recent_avg_grad_norm': np.mean([h['gradient_norm'] for h in recent_history]),
            'recent_avg_performance': np.mean([h['performance'] for h in recent_history]),
            'gradient_variance': np.var(self.gradient_stats['norms'][-100:]) if len(self.gradient_stats['norms']) >= 100 else 0.0
        }


class MAMLOptimizer(BaseMetaOptimizer):
    """
    MAML-специфичный оптимизатор
    
    MAML Optimization Strategy
    - Second-order gradient computation
    - Efficient memory management
    - Adaptive inner loop optimization
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: MetaOptimizerConfig,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(model, config, logger)
        
        # Создаем мета-оптимизатор
        self.meta_optimizer = self._create_optimizer()
        
        # Scheduler для мета-оптимизатора
        if config.use_scheduler:
            self.scheduler = self._create_scheduler()
        else:
            self.scheduler = None
        
        self.logger.info(f"MAMLOptimizer initialized with {config.optimizer_type}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Создает мета-оптимизатор"""
        if self.config.optimizer_type.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.meta_lr,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.meta_lr,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.meta_lr,
                momentum=self.config.beta1,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "rmsprop":
            return optim.RMSprop(
                self.model.parameters(),
                lr=self.config.meta_lr,
                alpha=self.config.beta2,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Создает scheduler для learning rate"""
        if self.config.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.meta_optimizer,
                T_max=1000,  # Максимальное количество эпох
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.meta_optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience
            )
        elif self.config.scheduler_type == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.meta_optimizer,
                gamma=self.config.scheduler_factor
            )
        elif self.config.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.meta_optimizer,
                step_size=50,
                gamma=self.config.scheduler_factor
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")
    
    def inner_loop_update(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        model_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Выполняет обновление во внутреннем цикле MAML"""
        
        # Копируем параметры
        updated_params = {}
        for name, param in model_params.items():
            updated_params[name] = param.clone().requires_grad_(True)
        
        # Внутренние шаги градиентного спуска
        for step in range(self.config.num_inner_steps):
            # Forward pass с текущими параметрами
            predictions = self._functional_forward(support_data, updated_params)
            
            # Вычисляем loss
            inner_loss = nn.functional.mse_loss(predictions, support_labels)
            
            # Вычисляем градиенты
            grads = torch.autograd.grad(
                inner_loss,
                updated_params.values(),
                create_graph=True,  # Для второго порядка
                allow_unused=True
            )
            
            # Обновляем параметры
            new_params = {}
            for (name, param), grad in zip(updated_params.items(), grads):
                if grad is not None:
                    new_params[name] = param - self.config.inner_lr * grad
                else:
                    new_params[name] = param
            
            updated_params = new_params
        
        return updated_params
    
    def _functional_forward(
        self,
        data: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Functional forward pass с заданными параметрами"""
        # Это упрощенная версия. В реальности нужен functional API
        # для модели или использование higher library
        
        # Временно заменяем параметры модели
        original_params = {}
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
            if name in params:
                param.data = params[name]
        
        try:
            output = self.model(data)
        finally:
            # Восстанавливаем параметры
            for name, param in self.model.named_parameters():
                param.data = original_params[name]
        
        return output
    
    def step(
        self,
        meta_loss: torch.Tensor,
        task_batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Выполняет один шаг MAML оптимизации"""
        
        # Обнуляем градиенты
        self.meta_optimizer.zero_grad()
        
        # Backward pass для мета-loss
        meta_loss.backward()
        
        # Обрезаем градиенты
        gradient_norm = self.clip_gradients()
        
        # Шаг оптимизации
        self.meta_optimizer.step()
        
        # Обновляем scheduler
        if self.scheduler and self.config.scheduler_type != "plateau":
            self.scheduler.step()
        elif self.scheduler and self.config.scheduler_type == "plateau":
            self.scheduler.step(meta_loss.item())
        
        # Обновляем статистики
        current_performance = meta_loss.item()
        self.update_learning_rate(current_performance)
        
        if current_performance < self.best_performance:
            self.best_performance = current_performance
        
        # Логируем
        self.log_optimization_step(
            meta_loss.item(),
            gradient_norm,
            current_performance
        )
        
        self.step_count += 1
        
        # Возвращаем метрики
        return {
            'meta_loss': meta_loss.item(),
            'gradient_norm': gradient_norm,
            'learning_rate': self.meta_optimizer.param_groups[0]['lr'],
            'step_count': self.step_count
        }


class ReptileOptimizer(BaseMetaOptimizer):
    """
    Reptile-специфичный оптимизатор
    
    First-Order Meta-Optimization
    - Simple parameter interpolation
    - Memory efficient
    - Fast convergence
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: MetaOptimizerConfig,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(model, config, logger)
        
        # Reptile не использует традиционный оптимизатор
        # Обновления делаются напрямую
        
        self.logger.info("ReptileOptimizer initialized")
    
    def inner_loop_update(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        model_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Выполняет обновление во внутреннем цикле Reptile"""
        
        # Создаем копию модели для обучения
        model_copy = type(self.model)(**self.model.get_config() if hasattr(self.model, 'get_config') else {})
        model_copy.load_state_dict(dict(self.model.named_parameters()))
        
        # Оптимизатор для внутреннего цикла
        inner_optimizer = optim.SGD(
            model_copy.parameters(),
            lr=self.config.inner_lr
        )
        
        # Внутренние шаги
        for step in range(self.config.num_inner_steps):
            inner_optimizer.zero_grad()
            
            predictions = model_copy(support_data)
            inner_loss = nn.functional.mse_loss(predictions, support_labels)
            
            inner_loss.backward()
            inner_optimizer.step()
        
        # Возвращаем обновленные параметры
        updated_params = {}
        for name, param in model_copy.named_parameters():
            updated_params[name] = param.data.clone()
        
        return updated_params
    
    def step(
        self,
        meta_loss: torch.Tensor,
        task_batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Выполняет один шаг Reptile оптимизации"""
        
        # Сохраняем исходные параметры
        original_params = {}
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
        
        # Собираем обновленные параметры от всех задач
        all_updated_params = []
        
        for task in task_batch:
            support_data = task['support_data']
            support_labels = task['support_labels']
            
            # Обновление для задачи
            updated_params = self.inner_loop_update(
                support_data, support_labels, original_params
            )
            all_updated_params.append(updated_params)
        
        # Reptile meta-update: интерполяция параметров
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in original_params:
                    # Вычисляем среднее направление обновления
                    param_updates = []
                    for updated_params in all_updated_params:
                        if name in updated_params:
                            update_direction = updated_params[name] - original_params[name]
                            param_updates.append(update_direction)
                    
                    if param_updates:
                        avg_update = torch.stack(param_updates).mean(dim=0)
                        
                        # Reptile update
                        param.data.add_(avg_update, alpha=self.current_lr)
                        
                        # Weight decay
                        if self.config.use_weight_decay:
                            param.data.mul_(1 - self.config.weight_decay * self.current_lr)
        
        # Вычисляем "gradient norm" для мониторинга
        total_update_norm = 0.0
        for updated_params in all_updated_params:
            for name, param in self.model.named_parameters():
                if name in updated_params:
                    update = updated_params[name] - original_params[name]
                    total_update_norm += torch.norm(update).item() ** 2
        
        gradient_norm = math.sqrt(total_update_norm)
        
        # Обновляем статистики
        current_performance = meta_loss.item()
        self.update_learning_rate(current_performance)
        
        if current_performance < self.best_performance:
            self.best_performance = current_performance
        
        # Логируем
        self.log_optimization_step(
            meta_loss.item(),
            gradient_norm,
            current_performance
        )
        
        self.step_count += 1
        
        return {
            'meta_loss': meta_loss.item(),
            'gradient_norm': gradient_norm,
            'learning_rate': self.current_lr,
            'step_count': self.step_count
        }


class AdaptiveMetaOptimizer(BaseMetaOptimizer):
    """
    Адаптивный мета-оптимизатор с автоматическим выбором стратегии
    
    Adaptive Optimization Strategy
    - Dynamic algorithm selection
    - Performance-based adaptation
    - Ensemble optimization
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: MetaOptimizerConfig,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(model, config, logger)
        
        # Создаем различные оптимизаторы
        self.optimizers = {
            'maml': MAMLOptimizer(model, config, logger),
            'reptile': ReptileOptimizer(model, config, logger)
        }
        
        # Статистики производительности каждого оптимизатора
        self.optimizer_performance = defaultdict(list)
        self.current_optimizer = 'maml'  # По умолчанию
        
        # Параметры адаптации
        self.adaptation_window = 20  # Окно для оценки производительности
        self.switch_threshold = 0.1  # Порог для переключения оптимизатора
        
        self.logger.info("AdaptiveMetaOptimizer initialized")
    
    def _evaluate_optimizer_performance(self) -> str:
        """Оценивает производительность оптимизаторов и выбирает лучший"""
        
        if len(self.optimizer_performance[self.current_optimizer]) < self.adaptation_window:
            return self.current_optimizer
        
        # Средняя производительность текущего оптимизатора
        current_performance = np.mean(
            self.optimizer_performance[self.current_optimizer][-self.adaptation_window:]
        )
        
        # Проверяем альтернативы
        best_optimizer = self.current_optimizer
        best_performance = current_performance
        
        for opt_name, opt in self.optimizers.items():
            if opt_name != self.current_optimizer and len(self.optimizer_performance[opt_name]) >= 5:
                alt_performance = np.mean(self.optimizer_performance[opt_name][-5:])
                
                if alt_performance < best_performance - self.switch_threshold:
                    best_optimizer = opt_name
                    best_performance = alt_performance
        
        if best_optimizer != self.current_optimizer:
            self.logger.info(f"Switching optimizer from {self.current_optimizer} to {best_optimizer}")
        
        return best_optimizer
    
    def inner_loop_update(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        model_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Делегирует внутреннее обновление текущему оптимизатору"""
        return self.optimizers[self.current_optimizer].inner_loop_update(
            support_data, support_labels, model_params
        )
    
    def step(
        self,
        meta_loss: torch.Tensor,
        task_batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Выполняет адаптивный шаг оптимизации"""
        
        # Выполняем шаг текущим оптимизатором
        step_metrics = self.optimizers[self.current_optimizer].step(meta_loss, task_batch)
        
        # Записываем производительность
        self.optimizer_performance[self.current_optimizer].append(meta_loss.item())
        
        # Оцениваем необходимость переключения
        new_optimizer = self._evaluate_optimizer_performance()
        if new_optimizer != self.current_optimizer:
            self.current_optimizer = new_optimizer
        
        # Обновляем общие статистики
        self.step_count += 1
        
        # Добавляем информацию о текущем оптимизаторе
        step_metrics['current_optimizer'] = self.current_optimizer
        step_metrics['optimizer_switches'] = len(set(self.optimizer_performance.keys()))
        
        return step_metrics
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Расширенная статистика с информацией об адаптации"""
        base_stats = super().get_optimization_statistics()
        
        # Статистики по оптимизаторам
        optimizer_stats = {}
        for opt_name, performance_history in self.optimizer_performance.items():
            if performance_history:
                optimizer_stats[opt_name] = {
                    'usage_count': len(performance_history),
                    'avg_performance': np.mean(performance_history),
                    'best_performance': np.min(performance_history),
                    'recent_performance': np.mean(performance_history[-10:]) if len(performance_history) >= 10 else np.mean(performance_history)
                }
        
        base_stats.update({
            'current_optimizer': self.current_optimizer,
            'optimizer_statistics': optimizer_stats
        })
        
        return base_stats


class MetaOptimizerFactory:
    """
    Factory для создания мета-оптимизаторов
    
    Optimizer Factory
    - Centralized optimizer creation
    - Configuration-based instantiation
    - Easy extensibility
    """
    
    @staticmethod
    def create_optimizer(
        optimizer_type: str,
        model: nn.Module,
        config: MetaOptimizerConfig,
        logger: Optional[logging.Logger] = None
    ) -> BaseMetaOptimizer:
        """
        Создает мета-оптимизатор заданного типа
        
        Args:
            optimizer_type: Тип оптимизатора (maml, reptile, adaptive)
            model: Модель для оптимизации
            config: Конфигурация оптимизатора
            logger: Логгер
            
        Returns:
            Экземпляр мета-оптимизатора
        """
        
        if optimizer_type.lower() == "maml":
            return MAMLOptimizer(model, config, logger)
        elif optimizer_type.lower() == "reptile":
            return ReptileOptimizer(model, config, logger)
        elif optimizer_type.lower() == "adaptive":
            return AdaptiveMetaOptimizer(model, config, logger)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    @staticmethod
    def get_default_config(optimizer_type: str) -> MetaOptimizerConfig:
        """Возвращает конфигурацию по умолчанию для оптимизатора"""
        
        if optimizer_type.lower() == "maml":
            return MetaOptimizerConfig(
                meta_lr=0.001,
                inner_lr=0.01,
                num_inner_steps=5,
                optimizer_type="adam",
                use_scheduler=True,
                scheduler_type="cosine"
            )
        elif optimizer_type.lower() == "reptile":
            return MetaOptimizerConfig(
                meta_lr=0.001,
                inner_lr=0.01,
                num_inner_steps=5,
                optimizer_type="sgd",
                use_scheduler=False,
                use_adaptive_lr=True
            )
        elif optimizer_type.lower() == "adaptive":
            return MetaOptimizerConfig(
                meta_lr=0.001,
                inner_lr=0.01,
                num_inner_steps=5,
                optimizer_type="adam",
                use_scheduler=True,
                use_adaptive_lr=True
            )
        else:
            return MetaOptimizerConfig()  # Базовая конфигурация