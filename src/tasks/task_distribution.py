"""
Task Distribution System
Scalable Task Management for Meta-Learning

Система распределения и управления задачами для мета-обучения в контексте
криптовалютного трейдинга. Поддерживает различные типы задач и распределений.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import random
from collections import defaultdict
import math

from ..utils.meta_utils import MetaLearningMetrics


@dataclass
class TaskConfig:
    """Конфигурация задачи для мета-обучения"""
    
    # Основные параметры
    num_classes: int = 5  # Количество классов (для классификации)
    num_support: int = 5  # Примеров на класс в support set
    num_query: int = 15   # Примеров на класс в query set
    
    # Тип задачи
    task_type: str = "classification"  # classification, regression, ranking
    
    # Сложность задачи
    difficulty_level: str = "medium"  # easy, medium, hard
    min_difficulty: float = 0.1
    max_difficulty: float = 1.0
    
    # Временные параметры
    time_horizon: int = 100  # Горизонт прогнозирования для временных рядов
    sequence_length: int = 50  # Длина входной последовательности
    
    # Маркетные параметры
    market_conditions: List[str] = field(default_factory=lambda: ["bull", "bear", "sideways"])
    volatility_range: Tuple[float, float] = (0.01, 0.1)  # Диапазон волатильности
    
    # Балансировка
    class_balance: str = "balanced"  # balanced, imbalanced, natural
    imbalance_ratio: float = 0.1  # Для несбалансированных классов
    
    # Качество данных
    noise_level: float = 0.05  # Уровень шума в данных
    missing_data_ratio: float = 0.0  # Доля пропущенных данных


@dataclass
class TaskMetadata:
    """Метаданные задачи"""
    
    task_id: str
    task_type: str
    difficulty: float
    source_domain: str  # crypto_pairs, market_regimes, strategies
    target_variable: str
    feature_names: List[str]
    data_quality_score: float
    created_timestamp: float
    
    # Crypto-specific
    trading_pair: Optional[str] = None
    exchange: Optional[str] = None
    timeframe: Optional[str] = None
    market_cap_category: Optional[str] = None  # large, mid, small, micro


class BaseTaskDistribution(ABC):
    """
    Базовый класс для распределения задач
    
    Abstract Task Distribution
    - Pluggable task generation
    - Consistent interface
    - Extensible architecture
    """
    
    def __init__(self, config: TaskConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.task_registry = {}
        self.metadata_registry = {}
    
    @abstractmethod
    def sample_task(self) -> Dict[str, torch.Tensor]:
        """Семплирует одну задачу"""
        pass
    
    @abstractmethod
    def sample_batch(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Семплирует batch задач"""
        pass
    
    @abstractmethod
    def get_task_difficulty(self, task_data: Dict[str, torch.Tensor]) -> float:
        """Оценивает сложность задачи"""
        pass
    
    def register_task(self, task_id: str, task_data: Dict[str, torch.Tensor], metadata: TaskMetadata):
        """Регистрирует задачу в реестре"""
        self.task_registry[task_id] = task_data
        self.metadata_registry[task_id] = metadata
        self.logger.debug(f"Registered task {task_id}")
    
    def get_task_by_id(self, task_id: str) -> Tuple[Dict[str, torch.Tensor], TaskMetadata]:
        """Получает задачу по ID"""
        if task_id not in self.task_registry:
            raise ValueError(f"Task {task_id} not found in registry")
        return self.task_registry[task_id], self.metadata_registry[task_id]
    
    def get_tasks_by_criteria(self, **criteria) -> List[Tuple[str, Dict[str, torch.Tensor], TaskMetadata]]:
        """Получает задачи по критериям"""
        matching_tasks = []
        
        for task_id, metadata in self.metadata_registry.items():
            match = True
            for key, value in criteria.items():
                if hasattr(metadata, key):
                    if isinstance(value, (list, tuple)):
                        if getattr(metadata, key) not in value:
                            match = False
                            break
                    else:
                        if getattr(metadata, key) != value:
                            match = False
                            break
                else:
                    match = False
                    break
            
            if match:
                matching_tasks.append((task_id, self.task_registry[task_id], metadata))
        
        return matching_tasks


class CryptoTaskDistribution(BaseTaskDistribution):
    """
    Распределение задач для криптовалютного трейдинга
    
    Domain-Specific Task Distribution
    - Crypto market simulation
    - Realistic market conditions
    - Multi-asset scenarios
    """
    
    def __init__(
        self,
        config: TaskConfig,
        crypto_data: Optional[Dict[str, np.ndarray]] = None,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, logger)
        
        self.crypto_data = crypto_data or self._generate_synthetic_data()
        self.available_pairs = list(self.crypto_data.keys())
        self.task_counter = 0
        
        # Статистики по задачам
        self.task_stats = defaultdict(int)
        
        self.logger.info(f"CryptoTaskDistribution initialized with {len(self.available_pairs)} trading pairs")
    
    def _generate_synthetic_data(self) -> Dict[str, np.ndarray]:
        """Генерирует синтетические данные криптовалют"""
        synthetic_data = {}
        
        # Основные торговые пары
        pairs = [
            "BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT",
            "SOLUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT", "LINKUSDT"
        ]
        
        for pair in pairs:
            # Генерируем OHLCV данные
            length = 10000
            base_price = np.random.uniform(0.1, 50000)
            
            # Геометрическое броуновское движение с трендом
            returns = np.random.normal(0.0001, 0.02, length)
            prices = base_price * np.exp(np.cumsum(returns))
            
            # OHLCV
            open_prices = prices
            high_prices = prices * (1 + np.abs(np.random.normal(0, 0.01, length)))
            low_prices = prices * (1 - np.abs(np.random.normal(0, 0.01, length)))
            close_prices = prices
            volumes = np.random.lognormal(10, 1, length)
            
            # Технические индикаторы
            rsi = np.random.uniform(20, 80, length)
            macd = np.random.normal(0, 0.1, length)
            bb_upper = high_prices * 1.02
            bb_lower = low_prices * 0.98
            
            # Объединяем в один массив
            data = np.column_stack([
                open_prices, high_prices, low_prices, close_prices, volumes,
                rsi, macd, bb_upper, bb_lower
            ])
            
            synthetic_data[pair] = data
        
        return synthetic_data
    
    def sample_task(self) -> Dict[str, torch.Tensor]:
        """Семплирует одну криптовалютную задачу"""
        task_id = f"crypto_task_{self.task_counter}"
        self.task_counter += 1
        
        # Выбираем торговую пару
        trading_pair = random.choice(self.available_pairs)
        data = self.crypto_data[trading_pair]
        
        # Определяем тип задачи
        if self.config.task_type == "classification":
            task_data, metadata = self._create_classification_task(task_id, trading_pair, data)
        elif self.config.task_type == "regression":
            task_data, metadata = self._create_regression_task(task_id, trading_pair, data)
        else:
            raise ValueError(f"Unsupported task type: {self.config.task_type}")
        
        # Регистрируем задачу
        self.register_task(task_id, task_data, metadata)
        self.task_stats[self.config.task_type] += 1
        
        return task_data
    
    def _create_classification_task(
        self,
        task_id: str,
        trading_pair: str,
        data: np.ndarray
    ) -> Tuple[Dict[str, torch.Tensor], TaskMetadata]:
        """Создает задачу классификации направления цены"""
        
        # Выбираем случайный период
        start_idx = random.randint(self.config.sequence_length, len(data) - self.config.time_horizon - 1000)
        end_idx = start_idx + 1000
        
        period_data = data[start_idx:end_idx]
        
        # Создаем features (sliding windows)
        features = []
        labels = []
        
        for i in range(self.config.sequence_length, len(period_data) - self.config.time_horizon):
            # Feature window
            feature_window = period_data[i-self.config.sequence_length:i, :]  # [seq_len, n_features]
            
            # Нормализация
            feature_window = (feature_window - feature_window.mean(axis=0)) / (feature_window.std(axis=0) + 1e-8)
            
            # Label: направление цены через time_horizon шагов
            current_price = period_data[i, 3]  # close price
            future_price = period_data[i + self.config.time_horizon, 3]
            
            # Классы: 0 - падение, 1 - рост, 2 - боковик
            price_change = (future_price - current_price) / current_price
            
            if price_change < -0.02:
                label = 0  # Падение
            elif price_change > 0.02:
                label = 1  # Рост
            else:
                label = 2  # Боковик
            
            features.append(feature_window.flatten())
            labels.append(label)
        
        features = np.array(features)
        labels = np.array(labels)
        
        # Фильтруем по доступным классам
        unique_labels = np.unique(labels)
        if len(unique_labels) < self.config.num_classes:
            # Если не хватает классов, дополняем случайными
            while len(unique_labels) < self.config.num_classes:
                fake_label = len(unique_labels)
                labels = np.append(labels, fake_label)
                features = np.vstack([features, features[-1]])
                unique_labels = np.unique(labels)
        
        # Выбираем нужное количество классов
        selected_classes = np.random.choice(unique_labels, self.config.num_classes, replace=False)
        
        # Support и query sets
        support_data, support_labels, query_data, query_labels = self._split_support_query(
            features, labels, selected_classes
        )
        
        task_data = {
            'support_data': torch.FloatTensor(support_data),
            'support_labels': torch.LongTensor(support_labels),
            'query_data': torch.FloatTensor(query_data),
            'query_labels': torch.LongTensor(query_labels)
        }
        
        # Метаданные
        metadata = TaskMetadata(
            task_id=task_id,
            task_type="classification",
            difficulty=self._compute_classification_difficulty(support_labels, query_labels),
            source_domain="crypto_pairs",
            target_variable="price_direction",
            feature_names=[f"feature_{i}" for i in range(features.shape[1])],
            data_quality_score=0.8,
            created_timestamp=0.0,
            trading_pair=trading_pair,
            timeframe="1h"
        )
        
        return task_data, metadata
    
    def _create_regression_task(
        self,
        task_id: str,
        trading_pair: str,
        data: np.ndarray
    ) -> Tuple[Dict[str, torch.Tensor], TaskMetadata]:
        """Создает задачу регрессии предсказания цены"""
        
        # Аналогично классификации, но с continuous targets
        start_idx = random.randint(self.config.sequence_length, len(data) - self.config.time_horizon - 1000)
        end_idx = start_idx + 1000
        
        period_data = data[start_idx:end_idx]
        
        features = []
        targets = []
        
        for i in range(self.config.sequence_length, len(period_data) - self.config.time_horizon):
            # Feature window
            feature_window = period_data[i-self.config.sequence_length:i, :]
            feature_window = (feature_window - feature_window.mean(axis=0)) / (feature_window.std(axis=0) + 1e-8)
            
            # Target: relative price change
            current_price = period_data[i, 3]
            future_price = period_data[i + self.config.time_horizon, 3]
            price_change = (future_price - current_price) / current_price
            
            features.append(feature_window.flatten())
            targets.append(price_change)
        
        features = np.array(features)
        targets = np.array(targets)
        
        # Случайно выбираем примеры для support и query
        n_total = len(features)
        n_support = self.config.num_support * self.config.num_classes  # Переиспользуем параметр
        n_query = self.config.num_query * self.config.num_classes
        
        indices = np.random.permutation(n_total)
        support_indices = indices[:n_support]
        query_indices = indices[n_support:n_support + n_query]
        
        task_data = {
            'support_data': torch.FloatTensor(features[support_indices]),
            'support_labels': torch.FloatTensor(targets[support_indices]),
            'query_data': torch.FloatTensor(features[query_indices]),
            'query_labels': torch.FloatTensor(targets[query_indices])
        }
        
        metadata = TaskMetadata(
            task_id=task_id,
            task_type="regression",
            difficulty=np.std(targets),  # Сложность = волатильность
            source_domain="crypto_pairs",
            target_variable="price_change",
            feature_names=[f"feature_{i}" for i in range(features.shape[1])],
            data_quality_score=0.8,
            created_timestamp=0.0,
            trading_pair=trading_pair,
            timeframe="1h"
        )
        
        return task_data, metadata
    
    def _split_support_query(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        selected_classes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Разделяет данные на support и query sets"""
        
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        
        for class_idx, class_label in enumerate(selected_classes):
            # Находим примеры этого класса
            class_mask = labels == class_label
            class_features = features[class_mask]
            class_labels = np.full(len(class_features), class_idx)  # Перенумеровываем классы
            
            if len(class_features) < self.config.num_support + self.config.num_query:
                # Дублируем примеры если не хватает
                needed = self.config.num_support + self.config.num_query
                indices = np.random.choice(len(class_features), needed, replace=True)
                class_features = class_features[indices]
                class_labels = np.full(needed, class_idx)
            
            # Случайно выбираем примеры
            indices = np.random.permutation(len(class_features))
            support_indices = indices[:self.config.num_support]
            query_indices = indices[self.config.num_support:self.config.num_support + self.config.num_query]
            
            support_data.append(class_features[support_indices])
            support_labels.append(class_labels[support_indices])
            query_data.append(class_features[query_indices])
            query_labels.append(class_labels[query_indices])
        
        return (
            np.vstack(support_data),
            np.concatenate(support_labels),
            np.vstack(query_data),
            np.concatenate(query_labels)
        )
    
    def _compute_classification_difficulty(
        self,
        support_labels: np.ndarray,
        query_labels: np.ndarray
    ) -> float:
        """Вычисляет сложность задачи классификации"""
        
        # Факторы сложности:
        # 1. Балансировка классов
        unique, counts = np.unique(support_labels, return_counts=True)
        balance_ratio = np.min(counts) / np.max(counts)
        
        # 2. Количество классов
        num_classes = len(unique)
        class_complexity = num_classes / 10.0  # Нормализуем
        
        # 3. Размер support set
        support_size_factor = 1.0 / len(support_labels) * 100
        
        # Общая сложность
        difficulty = (1.0 - balance_ratio) * 0.4 + class_complexity * 0.3 + support_size_factor * 0.3
        return np.clip(difficulty, 0.0, 1.0)
    
    def sample_batch(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Семплирует batch задач"""
        return [self.sample_task() for _ in range(batch_size)]
    
    def get_task_difficulty(self, task_data: Dict[str, torch.Tensor]) -> float:
        """Оценивает сложность задачи на основе данных"""
        support_labels = task_data['support_labels']
        query_labels = task_data['query_labels']
        
        if task_data['support_labels'].dtype == torch.long:
            # Классификация
            return self._compute_classification_difficulty(
                support_labels.numpy(), query_labels.numpy()
            )
        else:
            # Регрессия
            return float(torch.std(support_labels).item())
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по задачам"""
        return {
            'total_tasks': sum(self.task_stats.values()),
            'task_types': dict(self.task_stats),
            'available_pairs': self.available_pairs,
            'registered_tasks': len(self.task_registry)
        }


class CurriculumTaskDistribution(BaseTaskDistribution):
    """
    Распределение задач с curriculum learning
    
    Progressive Learning System
    - Adaptive difficulty progression
    - Performance-based task selection
    - Multi-objective optimization
    """
    
    def __init__(
        self,
        base_distribution: BaseTaskDistribution,
        config: TaskConfig,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, logger)
        
        self.base_distribution = base_distribution
        self.current_difficulty = config.min_difficulty
        self.performance_history = []
        self.difficulty_schedule = self._create_difficulty_schedule()
        
        # Параметры curriculum
        self.difficulty_increase_threshold = 0.8  # Accuracy threshold для увеличения сложности
        self.difficulty_decrease_threshold = 0.5  # Accuracy threshold для уменьшения сложности
        self.difficulty_step = 0.1
        self.patience = 5  # Количество эпох для изменения сложности
        
        self.logger.info(f"CurriculumTaskDistribution initialized with difficulty range: "
                        f"{config.min_difficulty}-{config.max_difficulty}")
    
    def _create_difficulty_schedule(self) -> List[float]:
        """Создает расписание увеличения сложности"""
        num_steps = 20
        min_diff = self.config.min_difficulty
        max_diff = self.config.max_difficulty
        
        # Экспоненциальное увеличение сложности
        schedule = []
        for i in range(num_steps):
            progress = i / (num_steps - 1)
            difficulty = min_diff + (max_diff - min_diff) * (progress ** 2)
            schedule.append(difficulty)
        
        return schedule
    
    def update_performance(self, performance_metrics: Dict[str, float]) -> None:
        """Обновляет историю производительности и адаптирует сложность"""
        self.performance_history.append(performance_metrics)
        
        # Оцениваем последние результаты
        if len(self.performance_history) >= self.patience:
            recent_performance = self.performance_history[-self.patience:]
            avg_accuracy = np.mean([p.get('accuracy', 0) for p in recent_performance])
            
            # Адаптируем сложность
            if avg_accuracy > self.difficulty_increase_threshold:
                # Увеличиваем сложность
                new_difficulty = min(
                    self.current_difficulty + self.difficulty_step,
                    self.config.max_difficulty
                )
                if new_difficulty > self.current_difficulty:
                    self.current_difficulty = new_difficulty
                    self.logger.info(f"Increased difficulty to {self.current_difficulty:.2f}")
            
            elif avg_accuracy < self.difficulty_decrease_threshold:
                # Уменьшаем сложность
                new_difficulty = max(
                    self.current_difficulty - self.difficulty_step,
                    self.config.min_difficulty
                )
                if new_difficulty < self.current_difficulty:
                    self.current_difficulty = new_difficulty
                    self.logger.info(f"Decreased difficulty to {self.current_difficulty:.2f}")
    
    def sample_task(self) -> Dict[str, torch.Tensor]:
        """Семплирует задачу с учетом текущей сложности"""
        # Генерируем несколько кандидатов и выбираем подходящий по сложности
        candidates = []
        difficulties = []
        
        for _ in range(10):  # Генерируем 10 кандидатов
            task = self.base_distribution.sample_task()
            difficulty = self.base_distribution.get_task_difficulty(task)
            candidates.append(task)
            difficulties.append(difficulty)
        
        # Выбираем задачу с ближайшей сложностью
        target_difficulty = self.current_difficulty
        best_idx = np.argmin([abs(d - target_difficulty) for d in difficulties])
        
        selected_task = candidates[best_idx]
        selected_difficulty = difficulties[best_idx]
        
        self.logger.debug(f"Selected task with difficulty {selected_difficulty:.3f} "
                         f"(target: {target_difficulty:.3f})")
        
        return selected_task
    
    def sample_batch(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Семплирует batch задач с учетом curriculum"""
        return [self.sample_task() for _ in range(batch_size)]
    
    def get_task_difficulty(self, task_data: Dict[str, torch.Tensor]) -> float:
        """Делегирует оценку сложности базовому распределению"""
        return self.base_distribution.get_task_difficulty(task_data)
    
    def get_curriculum_status(self) -> Dict[str, Any]:
        """Возвращает статус curriculum learning"""
        return {
            'current_difficulty': self.current_difficulty,
            'min_difficulty': self.config.min_difficulty,
            'max_difficulty': self.config.max_difficulty,
            'performance_history_length': len(self.performance_history),
            'recent_avg_performance': (
                np.mean([p.get('accuracy', 0) for p in self.performance_history[-5:]])
                if len(self.performance_history) >= 5 else 0.0
            )
        }


class MultiDomainTaskDistribution(BaseTaskDistribution):
    """
    Распределение задач из нескольких доменов
    
    Multi-Domain Meta-Learning
    - Cross-domain transfer
    - Domain adaptation
    - Balanced domain sampling
    """
    
    def __init__(
        self,
        domain_distributions: Dict[str, BaseTaskDistribution],
        config: TaskConfig,
        domain_weights: Optional[Dict[str, float]] = None,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, logger)
        
        self.domain_distributions = domain_distributions
        self.domain_names = list(domain_distributions.keys())
        
        # Веса доменов для семплирования
        if domain_weights is None:
            self.domain_weights = {name: 1.0 for name in self.domain_names}
        else:
            self.domain_weights = domain_weights
        
        # Нормализуем веса
        total_weight = sum(self.domain_weights.values())
        self.domain_weights = {
            name: weight / total_weight
            for name, weight in self.domain_weights.items()
        }
        
        # Статистики
        self.domain_stats = defaultdict(int)
        
        self.logger.info(f"MultiDomainTaskDistribution initialized with domains: {self.domain_names}")
    
    def sample_task(self) -> Dict[str, torch.Tensor]:
        """Семплирует задачу из случайного домена"""
        # Выбираем домен согласно весам
        domain_name = np.random.choice(
            self.domain_names,
            p=list(self.domain_weights.values())
        )
        
        # Семплируем задачу из выбранного домена
        task = self.domain_distributions[domain_name].sample_task()
        
        # Добавляем информацию о домене
        task['domain'] = domain_name
        
        self.domain_stats[domain_name] += 1
        return task
    
    def sample_batch(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Семплирует batch с балансировкой по доменам"""
        batch = []
        
        # Распределяем задачи по доменам
        domain_counts = {}
        remaining = batch_size
        
        for domain_name in self.domain_names[:-1]:  # Все кроме последнего
            count = int(batch_size * self.domain_weights[domain_name])
            domain_counts[domain_name] = count
            remaining -= count
        
        # Остаток отдаем последнему домену
        domain_counts[self.domain_names[-1]] = remaining
        
        # Семплируем задачи
        for domain_name, count in domain_counts.items():
            if count > 0:
                domain_tasks = self.domain_distributions[domain_name].sample_batch(count)
                for task in domain_tasks:
                    task['domain'] = domain_name
                batch.extend(domain_tasks)
        
        # Перемешиваем batch
        random.shuffle(batch)
        return batch
    
    def get_task_difficulty(self, task_data: Dict[str, torch.Tensor]) -> float:
        """Оценивает сложность задачи через соответствующий домен"""
        domain_name = task_data.get('domain', self.domain_names[0])
        return self.domain_distributions[domain_name].get_task_difficulty(task_data)
    
    def update_domain_weights(self, performance_by_domain: Dict[str, float]) -> None:
        """Обновляет веса доменов на основе производительности"""
        # Увеличиваем веса доменов с низкой производительностью
        for domain_name in self.domain_names:
            performance = performance_by_domain.get(domain_name, 0.5)
            # Обратная зависимость: чем хуже производительность, тем больше вес
            self.domain_weights[domain_name] = 1.0 / (performance + 0.1)
        
        # Нормализуем веса
        total_weight = sum(self.domain_weights.values())
        self.domain_weights = {
            name: weight / total_weight
            for name, weight in self.domain_weights.items()
        }
        
        self.logger.info(f"Updated domain weights: {self.domain_weights}")
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по доменам"""
        return {
            'domain_weights': self.domain_weights,
            'domain_stats': dict(self.domain_stats),
            'total_tasks': sum(self.domain_stats.values())
        }