"""
Inner Loop Optimization
Task-Specific Optimization for Meta-Learning

Оптимизация внутреннего цикла для быстрой адаптации к новым задачам
с различными стратегиями и алгоритмами оптимизации.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
import logging
import numpy as np
from abc import ABC, abstractmethod
import copy
import math
from collections import OrderedDict, defaultdict

from ..utils.gradient_utils import GradientManager


@dataclass
class InnerLoopConfig:
    """Конфигурация для внутреннего цикла оптимизации"""
    
    # Основные параметры
    learning_rate: float = 0.01  # Скорость обучения во внутреннем цикле
    num_steps: int = 5  # Количество шагов оптимизации
    
    # Оптимизатор
    optimizer_type: str = "sgd"  # sgd, adam, adamw, rmsprop
    momentum: float = 0.0  # Momentum для SGD
    beta1: float = 0.9  # Beta1 для Adam-like оптимизаторов
    beta2: float = 0.999  # Beta2 для Adam-like оптимизаторов
    
    # Регуляризация
    weight_decay: float = 0.0  # L2 регуляризация
    grad_clip: Optional[float] = None  # Обрезка градиентов
    
    # Адаптивные стратегии
    use_adaptive_lr: bool = False  # Адаптивная скорость обучения
    lr_decay_factor: float = 0.95  # Фактор уменьшения LR
    min_lr: float = 1e-5  # Минимальная скорость обучения
    
    # Stopping criteria
    early_stopping: bool = False  # Ранняя остановка
    patience: int = 3  # Терпение для ранней остановки
    min_improvement: float = 1e-4  # Минимальное улучшение
    
    # Advanced features
    use_meta_initialization: bool = False  # Использование мета-инициализации
    use_gradient_modification: bool = False  # Модификация градиентов
    use_loss_scaling: bool = False  # Масштабирование loss
    
    # Специфичные для крипто
    market_aware_lr: bool = False  # Учет волатильности рынка
    volatility_adjustment: float = 1.0  # Корректировка на волатильность


class BaseInnerLoopOptimizer(ABC):
    """
    Базовый класс для оптимизаторов внутреннего цикла
    
    Abstract Inner Loop Optimizer
    - Task-specific adaptation
    - Efficient parameter updates
    - Performance monitoring
    """
    
    def __init__(
        self,
        config: InnerLoopConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Статистики
        self.adaptation_history = []
        self.gradient_stats = defaultdict(list)
        
        # Утилиты
        self.gradient_manager = GradientManager()
    
    @abstractmethod
    def adapt(
        self,
        model: nn.Module,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        loss_fn: Optional[Callable] = None
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """
        Адаптирует модель к задаче
        
        Args:
            model: Модель для адаптации
            support_data: Support данные
            support_labels: Support метки
            loss_fn: Функция потерь (по умолчанию MSE)
            
        Returns:
            Адаптированная модель и метрики адаптации
        """
        pass
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """Вычисляет loss с возможными модификациями"""
        if loss_fn is None:
            loss_fn = F.mse_loss
        
        base_loss = loss_fn(predictions, targets)
        
        # Loss scaling для стабильности
        if self.config.use_loss_scaling:
            loss_scale = self._compute_loss_scale(predictions, targets)
            base_loss = base_loss * loss_scale
        
        return base_loss
    
    def _compute_loss_scale(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """Вычисляет масштабирующий фактор для loss"""
        # Адаптивное масштабирование на основе величины loss
        with torch.no_grad():
            loss_magnitude = F.mse_loss(predictions, targets).item()
            
            if loss_magnitude > 1.0:
                scale = 1.0 / math.sqrt(loss_magnitude)
            elif loss_magnitude < 0.01:
                scale = math.sqrt(100 * loss_magnitude)
            else:
                scale = 1.0
        
        return scale
    
    def _adjust_learning_rate_for_market(
        self,
        base_lr: float,
        support_data: torch.Tensor
    ) -> float:
        """Корректирует LR на основе волатильности рынка"""
        if not self.config.market_aware_lr:
            return base_lr
        
        # Оцениваем волатильность из данных
        with torch.no_grad():
            # Предполагаем, что последние столбцы - это ценовые данные
            if len(support_data.shape) > 1 and support_data.shape[1] > 1:
                price_data = support_data[:, -1]  # Последний столбец как цена
                if len(price_data) > 1:
                    returns = torch.diff(price_data)
                    volatility = torch.std(returns).item()
                    
                    # Корректируем LR: больше волатильность -> меньше LR
                    volatility_factor = 1.0 / (1.0 + volatility * self.config.volatility_adjustment)
                    return base_lr * volatility_factor
        
        return base_lr
    
    def log_adaptation_step(
        self,
        step: int,
        loss: float,
        gradient_norm: float,
        learning_rate: float
    ) -> None:
        """Логирует шаг адаптации"""
        step_info = {
            'step': step,
            'loss': loss,
            'gradient_norm': gradient_norm,
            'learning_rate': learning_rate
        }
        
        self.adaptation_history.append(step_info)
        self.gradient_stats['norms'].append(gradient_norm)
        self.gradient_stats['losses'].append(loss)


class SGDInnerLoopOptimizer(BaseInnerLoopOptimizer):
    """
    SGD-based оптимизатор для внутреннего цикла
    
    SGD Inner Loop Optimization
    - Simple and efficient
    - Momentum support
    - Adaptive learning rate
    """
    
    def adapt(
        self,
        model: nn.Module,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        loss_fn: Optional[Callable] = None
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """Адаптирует модель с помощью SGD"""
        
        # Создаем копию модели
        adapted_model = copy.deepcopy(model)
        adapted_model.train()
        
        # Создаем оптимизатор
        base_lr = self._adjust_learning_rate_for_market(
            self.config.learning_rate, support_data
        )
        
        optimizer = optim.SGD(
            adapted_model.parameters(),
            lr=base_lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        # Адаптивная скорость обучения
        current_lr = base_lr
        
        # Ранняя остановка
        best_loss = float('inf')
        patience_counter = 0
        
        # Метрики адаптации
        adaptation_losses = []
        gradient_norms = []
        
        # Очищаем историю адаптации
        self.adaptation_history.clear()
        
        for step in range(self.config.num_steps):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = adapted_model(support_data)
            loss = self.compute_loss(predictions, support_labels, loss_fn)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            gradient_norm = 0.0
            if self.config.grad_clip is not None:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    adapted_model.parameters(), self.config.grad_clip
                ).item()
            else:
                gradient_norm = self.gradient_manager.compute_gradient_norm(
                    adapted_model.parameters()
                )
            
            # Адаптивная корректировка LR
            if self.config.use_adaptive_lr and step > 0:
                if loss.item() > adaptation_losses[-1]:
                    current_lr = max(
                        current_lr * self.config.lr_decay_factor,
                        self.config.min_lr
                    )
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
            
            # Optimization step
            optimizer.step()
            
            # Логирование
            loss_value = loss.item()
            adaptation_losses.append(loss_value)
            gradient_norms.append(gradient_norm)
            
            self.log_adaptation_step(step, loss_value, gradient_norm, current_lr)
            
            # Ранняя остановка
            if self.config.early_stopping:
                if loss_value < best_loss - self.config.min_improvement:
                    best_loss = loss_value
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.patience:
                    self.logger.debug(f"Early stopping at step {step}")
                    break
        
        # Метрики адаптации
        metrics = {
            'final_loss': adaptation_losses[-1] if adaptation_losses else float('inf'),
            'initial_loss': adaptation_losses[0] if adaptation_losses else float('inf'),
            'loss_improvement': (adaptation_losses[0] - adaptation_losses[-1]) if len(adaptation_losses) > 1 else 0.0,
            'avg_gradient_norm': np.mean(gradient_norms) if gradient_norms else 0.0,
            'final_lr': current_lr,
            'num_steps': len(adaptation_losses)
        }
        
        return adapted_model, metrics


class AdamInnerLoopOptimizer(BaseInnerLoopOptimizer):
    """
    Adam-based оптимизатор для внутреннего цикла
    
    Adam Inner Loop Optimization
    - Adaptive moment estimation
    - Better convergence properties
    - Suitable for noisy gradients
    """
    
    def adapt(
        self,
        model: nn.Module,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        loss_fn: Optional[Callable] = None
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """Адаптирует модель с помощью Adam"""
        
        # Создаем копию модели
        adapted_model = copy.deepcopy(model)
        adapted_model.train()
        
        # Создаем оптимизатор
        base_lr = self._adjust_learning_rate_for_market(
            self.config.learning_rate, support_data
        )
        
        optimizer = optim.Adam(
            adapted_model.parameters(),
            lr=base_lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay
        )
        
        # Метрики адаптации
        adaptation_losses = []
        gradient_norms = []
        
        # Очищаем историю
        self.adaptation_history.clear()
        
        for step in range(self.config.num_steps):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = adapted_model(support_data)
            loss = self.compute_loss(predictions, support_labels, loss_fn)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            gradient_norm = 0.0
            if self.config.grad_clip is not None:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    adapted_model.parameters(), self.config.grad_clip
                ).item()
            else:
                gradient_norm = self.gradient_manager.compute_gradient_norm(
                    adapted_model.parameters()
                )
            
            # Optimization step
            optimizer.step()
            
            # Логирование
            loss_value = loss.item()
            adaptation_losses.append(loss_value)
            gradient_norms.append(gradient_norm)
            
            current_lr = optimizer.param_groups[0]['lr']
            self.log_adaptation_step(step, loss_value, gradient_norm, current_lr)
        
        # Метрики
        metrics = {
            'final_loss': adaptation_losses[-1] if adaptation_losses else float('inf'),
            'initial_loss': adaptation_losses[0] if adaptation_losses else float('inf'),
            'loss_improvement': (adaptation_losses[0] - adaptation_losses[-1]) if len(adaptation_losses) > 1 else 0.0,
            'avg_gradient_norm': np.mean(gradient_norms) if gradient_norms else 0.0,
            'final_lr': base_lr,
            'num_steps': len(adaptation_losses)
        }
        
        return adapted_model, metrics


class MetaInitializedInnerLoopOptimizer(BaseInnerLoopOptimizer):
    """
    Оптимизатор с мета-инициализацией параметров
    
    Meta-Initialized Optimization
    - Better initialization from meta-learning
    - Faster convergence
    - Task-specific parameter scaling
    """
    
    def __init__(
        self,
        config: InnerLoopConfig,
        meta_parameters: Optional[Dict[str, torch.Tensor]] = None,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, logger)
        
        self.meta_parameters = meta_parameters or {}
        
        # Параметры мета-инициализации
        self.initialization_scale = 0.1  # Масштаб инициализации
        self.parameter_adaptation_rate = 0.1  # Скорость адаптации параметров
    
    def _initialize_with_meta_parameters(self, model: nn.Module) -> None:
        """Инициализирует модель с мета-параметрами"""
        if not self.config.use_meta_initialization or not self.meta_parameters:
            return
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.meta_parameters:
                    # Комбинируем текущие параметры с мета-параметрами
                    meta_param = self.meta_parameters[name]
                    
                    # Взвешенная комбинация
                    param.data = (
                        (1 - self.parameter_adaptation_rate) * param.data +
                        self.parameter_adaptation_rate * meta_param
                    )
    
    def adapt(
        self,
        model: nn.Module,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        loss_fn: Optional[Callable] = None
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """Адаптирует модель с мета-инициализацией"""
        
        # Создаем копию модели
        adapted_model = copy.deepcopy(model)
        
        # Мета-инициализация
        self._initialize_with_meta_parameters(adapted_model)
        
        adapted_model.train()
        
        # Используем адаптивный выбор оптимизатора
        base_lr = self._adjust_learning_rate_for_market(
            self.config.learning_rate, support_data
        )
        
        if self.config.optimizer_type.lower() == "adam":
            optimizer = optim.Adam(
                adapted_model.parameters(),
                lr=base_lr,
                betas=(self.config.beta1, self.config.beta2),
                weight_decay=self.config.weight_decay
            )
        else:
            optimizer = optim.SGD(
                adapted_model.parameters(),
                lr=base_lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        
        # Метрики
        adaptation_losses = []
        gradient_norms = []
        parameter_changes = []
        
        # Сохраняем начальные параметры для отслеживания изменений
        initial_params = {}
        for name, param in adapted_model.named_parameters():
            initial_params[name] = param.data.clone()
        
        self.adaptation_history.clear()
        
        for step in range(self.config.num_steps):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = adapted_model(support_data)
            loss = self.compute_loss(predictions, support_labels, loss_fn)
            
            # Backward pass
            loss.backward()
            
            # Модификация градиентов если включена
            if self.config.use_gradient_modification:
                self._modify_gradients(adapted_model, step)
            
            # Gradient clipping
            gradient_norm = 0.0
            if self.config.grad_clip is not None:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    adapted_model.parameters(), self.config.grad_clip
                ).item()
            else:
                gradient_norm = self.gradient_manager.compute_gradient_norm(
                    adapted_model.parameters()
                )
            
            # Optimization step
            optimizer.step()
            
            # Отслеживаем изменения параметров
            total_param_change = 0.0
            for name, param in adapted_model.named_parameters():
                if name in initial_params:
                    change = torch.norm(param.data - initial_params[name]).item()
                    total_param_change += change
            
            # Логирование
            loss_value = loss.item()
            adaptation_losses.append(loss_value)
            gradient_norms.append(gradient_norm)
            parameter_changes.append(total_param_change)
            
            current_lr = optimizer.param_groups[0]['lr']
            self.log_adaptation_step(step, loss_value, gradient_norm, current_lr)
        
        # Расширенные метрики
        metrics = {
            'final_loss': adaptation_losses[-1] if adaptation_losses else float('inf'),
            'initial_loss': adaptation_losses[0] if adaptation_losses else float('inf'),
            'loss_improvement': (adaptation_losses[0] - adaptation_losses[-1]) if len(adaptation_losses) > 1 else 0.0,
            'avg_gradient_norm': np.mean(gradient_norms) if gradient_norms else 0.0,
            'final_lr': base_lr,
            'num_steps': len(adaptation_losses),
            'total_parameter_change': parameter_changes[-1] if parameter_changes else 0.0,
            'parameter_change_rate': np.mean(np.diff(parameter_changes)) if len(parameter_changes) > 1 else 0.0,
            'convergence_rate': self._compute_convergence_rate(adaptation_losses)
        }
        
        return adapted_model, metrics
    
    def _modify_gradients(self, model: nn.Module, step: int) -> None:
        """Модифицирует градиенты для лучшей адаптации"""
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    # Gradient scaling на основе шага
                    scale_factor = 1.0 / (1.0 + 0.1 * step)
                    param.grad.data *= scale_factor
                    
                    # Добавляем небольшой шум для регуляризации
                    noise = torch.randn_like(param.grad) * 0.001
                    param.grad.data += noise
    
    def _compute_convergence_rate(self, losses: List[float]) -> float:
        """Вычисляет скорость сходимости"""
        if len(losses) < 2:
            return 0.0
        
        # Экспоненциальная скорость сходимости
        improvements = []
        for i in range(1, len(losses)):
            if losses[i-1] > 0:
                improvement = (losses[i-1] - losses[i]) / losses[i-1]
                improvements.append(max(0, improvement))
        
        return np.mean(improvements) if improvements else 0.0
    
    def update_meta_parameters(self, new_meta_parameters: Dict[str, torch.Tensor]) -> None:
        """Обновляет мета-параметры"""
        self.meta_parameters = new_meta_parameters
        self.logger.debug(f"Updated meta-parameters for {len(new_meta_parameters)} parameters")


class InnerLoopOptimizerFactory:
    """
    Factory для создания оптимизаторов внутреннего цикла
    
    Inner Loop Optimizer Factory
    - Centralized optimizer creation
    - Configuration-based instantiation
    - Support for different algorithms
    """
    
    @staticmethod
    def create_optimizer(
        optimizer_type: str,
        config: InnerLoopConfig,
        meta_parameters: Optional[Dict[str, torch.Tensor]] = None,
        logger: Optional[logging.Logger] = None
    ) -> BaseInnerLoopOptimizer:
        """
        Создает оптимизатор внутреннего цикла
        
        Args:
            optimizer_type: Тип оптимизатора (sgd, adam, meta_init)
            config: Конфигурация оптимизатора
            meta_parameters: Мета-параметры для инициализации
            logger: Логгер
            
        Returns:
            Экземпляр оптимизатора
        """
        
        if optimizer_type.lower() == "sgd":
            return SGDInnerLoopOptimizer(config, logger)
        elif optimizer_type.lower() == "adam":
            return AdamInnerLoopOptimizer(config, logger)
        elif optimizer_type.lower() == "meta_init":
            return MetaInitializedInnerLoopOptimizer(config, meta_parameters, logger)
        else:
            raise ValueError(f"Unknown inner loop optimizer type: {optimizer_type}")
    
    @staticmethod
    def get_default_config(optimizer_type: str) -> InnerLoopConfig:
        """Возвращает конфигурацию по умолчанию"""
        
        if optimizer_type.lower() == "sgd":
            return InnerLoopConfig(
                learning_rate=0.01,
                num_steps=5,
                optimizer_type="sgd",
                momentum=0.9,
                use_adaptive_lr=True
            )
        elif optimizer_type.lower() == "adam":
            return InnerLoopConfig(
                learning_rate=0.001,
                num_steps=5,
                optimizer_type="adam",
                beta1=0.9,
                beta2=0.999,
                use_adaptive_lr=False
            )
        elif optimizer_type.lower() == "meta_init":
            return InnerLoopConfig(
                learning_rate=0.01,
                num_steps=3,  # Меньше шагов благодаря лучшей инициализации
                optimizer_type="adam",
                use_meta_initialization=True,
                use_gradient_modification=True,
                early_stopping=True
            )
        else:
            return InnerLoopConfig()


class AdaptiveInnerLoopManager:
    """
    Менеджер для адаптивного выбора стратегии внутреннего цикла
    
    Adaptive Inner Loop Strategy
    - Performance-based optimizer selection
    - Dynamic strategy switching
    - Task-specific adaptation
    """
    
    def __init__(
        self,
        base_config: InnerLoopConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.base_config = base_config
        self.logger = logger or logging.getLogger(__name__)
        
        # Доступные оптимизаторы
        self.optimizers = {
            'sgd': InnerLoopOptimizerFactory.create_optimizer('sgd', base_config, None, logger),
            'adam': InnerLoopOptimizerFactory.create_optimizer('adam', base_config, None, logger)
        }
        
        # Статистики производительности
        self.performance_history = defaultdict(list)
        self.current_optimizer = 'adam'  # По умолчанию
        
        # Параметры адаптации
        self.evaluation_window = 10
        self.switch_threshold = 0.05  # 5% улучшение для переключения
        
    def adapt_with_best_optimizer(
        self,
        model: nn.Module,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        loss_fn: Optional[Callable] = None
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """Адаптирует модель с лучшим оптимизатором"""
        
        # Выбираем лучший оптимизатор
        best_optimizer_name = self._select_best_optimizer()
        
        # Адаптируем
        adapted_model, metrics = self.optimizers[best_optimizer_name].adapt(
            model, support_data, support_labels, loss_fn
        )
        
        # Записываем производительность
        self.performance_history[best_optimizer_name].append(metrics['final_loss'])
        
        # Добавляем информацию об использованном оптимизаторе
        metrics['optimizer_used'] = best_optimizer_name
        
        return adapted_model, metrics
    
    def _select_best_optimizer(self) -> str:
        """Выбирает лучший оптимизатор на основе истории"""
        
        # Если недостаточно данных, используем текущий
        if len(self.performance_history[self.current_optimizer]) < self.evaluation_window:
            return self.current_optimizer
        
        # Оцениваем производительность всех оптимизаторов
        optimizer_scores = {}
        
        for opt_name in self.optimizers.keys():
            if len(self.performance_history[opt_name]) >= 5:  # Минимум данных
                recent_performance = self.performance_history[opt_name][-5:]
                optimizer_scores[opt_name] = np.mean(recent_performance)
        
        # Если нет альтернатив, используем текущий
        if len(optimizer_scores) <= 1:
            return self.current_optimizer
        
        # Выбираем лучший
        best_optimizer = min(optimizer_scores.keys(), key=lambda x: optimizer_scores[x])
        
        # Переключаемся только если значительное улучшение
        current_score = optimizer_scores.get(self.current_optimizer, float('inf'))
        best_score = optimizer_scores[best_optimizer]
        
        if (current_score - best_score) / current_score > self.switch_threshold:
            self.logger.info(f"Switching inner loop optimizer from {self.current_optimizer} to {best_optimizer}")
            self.current_optimizer = best_optimizer
        
        return self.current_optimizer
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Возвращает статистики производительности"""
        stats = {}
        
        for opt_name, performance_list in self.performance_history.items():
            if performance_list:
                stats[opt_name] = {
                    'usage_count': len(performance_list),
                    'avg_loss': np.mean(performance_list),
                    'best_loss': np.min(performance_list),
                    'recent_avg_loss': np.mean(performance_list[-10:]) if len(performance_list) >= 10 else np.mean(performance_list)
                }
        
        stats['current_optimizer'] = self.current_optimizer
        return stats