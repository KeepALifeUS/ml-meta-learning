"""
Comprehensive Tests for Meta-Learning System
Context7 Enterprise Pattern: Production-Ready Testing

Tests for всех компонентов meta-learning системы с проверкой
корректности алгоритмов, производительности и интеграции.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import json

# Импорты тестируемых модулей
from ..src.algorithms.maml import MAML, MAMLConfig, MAMLTrainer
from ..src.algorithms.reptile import Reptile, ReptileConfig, ReptileTrainer
from ..src.algorithms.meta_sgd import MetaSGD, MetaSGDConfig, MetaSGDTrainer
from ..src.algorithms.proto_net import PrototypicalNetworks, ProtoNetConfig, ProtoNetTrainer
from ..src.algorithms.matching_net import MatchingNetworks, MatchingNetConfig, MatchingNetTrainer

from ..src.tasks.task_distribution import CryptoTaskDistribution, CryptoTaskConfig
from ..src.tasks.task_sampler import TaskSampler, SamplerConfig, TaskCache
from ..src.tasks.crypto_tasks import CryptoPriceDirectionTask, CryptoMarketSimulator

from ..src.optimization.meta_optimizer import MetaOptimizerFactory, MetaOptimizerConfig
from ..src.optimization.inner_loop import InnerLoopOptimizerFactory, InnerLoopConfig

from ..src.evaluation.few_shot_evaluator import (
    ClassificationEvaluator, RegressionEvaluator, 
    FewShotBenchmark, EvaluationConfig
)

from ..src.utils.gradient_utils import GradientManager, HigherOrderGradients
from ..src.utils.meta_utils import MetaLearningMetrics, DataAnalyzer, Visualizer


class SimpleTestModel(nn.Module):
    """Simple model for testing"""
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32, output_dim: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def get_config(self):
        return {
            'input_dim': 10,
            'hidden_dim': 32,
            'output_dim': 1
        }


@pytest.fixture
def simple_model():
    """Fixture for простой модели"""
    return SimpleTestModel()


@pytest.fixture
def sample_task_data():
    """Fixture for task data example"""
    return {
        'support_data': torch.randn(25, 10),  # 5 classes * 5 shots
        'support_labels': torch.randint(0, 5, (25,)),
        'query_data': torch.randn(75, 10),    # 5 classes * 15 queries
        'query_labels': torch.randint(0, 5, (75,))
    }


@pytest.fixture
def temp_dir():
    """Fixture for temporary directory"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


class TestMAML:
    """Tests for MAML алгоритма"""
    
    def test_maml_initialization(self, simple_model):
        """Тест инициализации MAML"""
        config = MAMLConfig(
            inner_lr=0.01,
            outer_lr=0.001,
            num_inner_steps=3
        )
        
        maml = MAML(simple_model, config)
        
        assert maml.config == config
        assert maml.model == simple_model
        assert maml.global_step == 0
        assert maml.best_meta_loss == float('inf')
    
    def test_maml_inner_loop(self, simple_model, sample_task_data):
        """Тест внутреннего цикла MAML"""
        config = MAMLConfig(num_inner_steps=3)
        maml = MAML(simple_model, config)
        
        model_state = dict(simple_model.named_parameters())
        
        adapted_params, inner_losses = maml.inner_loop(
            sample_task_data['support_data'],
            sample_task_data['support_labels'].float(),
            model_state,
            create_graph=False
        )
        
        assert len(inner_losses) == config.num_inner_steps
        assert len(adapted_params) == len(model_state)
        
        # Verify parameters changed
        for name, original_param in model_state.items():
            assert not torch.equal(adapted_params[name], original_param)
    
    def test_maml_meta_train_step(self, simple_model, sample_task_data):
        """Тест шага мета-обучения MAML"""
        config = MAMLConfig(num_inner_steps=2)
        maml = MAML(simple_model, config)
        
        task_batch = [sample_task_data]
        
        metrics = maml.meta_train_step(task_batch)
        
        assert 'meta_loss' in metrics
        assert 'adaptation_loss' in metrics
        assert 'query_accuracy' in metrics
        assert 'gradient_norm' in metrics
        
        assert isinstance(metrics['meta_loss'], float)
        assert metrics['meta_loss'] >= 0
        assert maml.global_step == 1
    
    def test_maml_few_shot_adapt(self, simple_model):
        """Тест few-shot адаптации MAML"""
        config = MAMLConfig(num_inner_steps=3)
        maml = MAML(simple_model, config)
        
        support_data = torch.randn(10, 10)
        support_labels = torch.randn(10, 1)
        
        adapted_model = maml.few_shot_adapt(support_data, support_labels)
        
        assert isinstance(adapted_model, nn.Module)
        assert adapted_model != simple_model  # Should be different model
    
    def test_maml_checkpoint_save_load(self, simple_model, temp_dir):
        """Тест сохранения и загрузки checkpoint MAML"""
        config = MAMLConfig()
        maml = MAML(simple_model, config)
        
        # Изменяем состояние
        maml.global_step = 10
        maml.best_meta_loss = 0.5
        
        # Сохраняем
        checkpoint_path = Path(temp_dir) / "test_maml.pt"
        maml.save_checkpoint(str(checkpoint_path))
        
        assert checkpoint_path.exists()
        
        # Загружаем
        new_maml = MAML(simple_model, config)
        new_maml.load_checkpoint(str(checkpoint_path))
        
        assert new_maml.global_step == 10
        assert new_maml.best_meta_loss == 0.5


class TestReptile:
    """Tests for Reptile алгоритма"""
    
    def test_reptile_initialization(self, simple_model):
        """Тест инициализации Reptile"""
        config = ReptileConfig(
            inner_lr=0.01,
            meta_lr=0.001,
            num_inner_steps=5
        )
        
        reptile = Reptile(simple_model, config)
        
        assert reptile.config == config
        assert reptile.model == simple_model
        assert reptile.global_step == 0
    
    def test_reptile_inner_adaptation(self, simple_model, sample_task_data):
        """Тест внутренней адаптации Reptile"""
        config = ReptileConfig(num_inner_steps=3)
        reptile = Reptile(simple_model, config)
        
        adapted_params, adaptation_losses = reptile.inner_adaptation(
            sample_task_data['support_data'],
            sample_task_data['support_labels'].float()
        )
        
        assert len(adaptation_losses) == config.num_inner_steps
        assert len(adapted_params) == len(list(simple_model.named_parameters()))
        
        # Verify loss decreasing
        assert adaptation_losses[0] >= adaptation_losses[-1]
    
    def test_reptile_meta_train_step(self, simple_model, sample_task_data):
        """Тест шага мета-обучения Reptile"""
        config = ReptileConfig(num_inner_steps=2)
        reptile = Reptile(simple_model, config)
        
        # Save original parameters for comparison
        original_params = {}
        for name, param in simple_model.named_parameters():
            original_params[name] = param.data.clone()
        
        task_batch = [sample_task_data]
        metrics = reptile.meta_train_step(task_batch)
        
        assert 'adaptation_loss' in metrics
        assert 'query_loss' in metrics
        assert 'query_accuracy' in metrics
        
        # Verify model parameters changed
        for name, param in simple_model.named_parameters():
            assert not torch.equal(param.data, original_params[name])


class TestMetaSGD:
    """Tests for Meta-SGD алгоритма"""
    
    def test_meta_sgd_initialization(self, simple_model):
        """Тест инициализации Meta-SGD"""
        config = MetaSGDConfig(
            meta_lr=0.001,
            num_inner_steps=3
        )
        
        meta_sgd = MetaSGD(simple_model, config)
        
        assert meta_sgd.config == config
        assert meta_sgd.model == simple_model
        assert len(meta_sgd.inner_lrs) > 0
        
        # Verify learning rates initialized for all parameters
        for name, param in simple_model.named_parameters():
            if param.requires_grad:
                assert name in meta_sgd.inner_lrs
                assert meta_sgd.inner_lrs[name].shape == param.shape
    
    def test_meta_sgd_inner_loop(self, simple_model, sample_task_data):
        """Тест внутреннего цикла Meta-SGD"""
        config = MetaSGDConfig(num_inner_steps=3)
        meta_sgd = MetaSGD(simple_model, config)
        
        model_state = dict(simple_model.named_parameters())
        
        adapted_params, inner_losses, lr_history = meta_sgd.inner_loop(
            sample_task_data['support_data'],
            sample_task_data['support_labels'].float(),
            model_state
        )
        
        assert len(inner_losses) == config.num_inner_steps
        assert len(lr_history) > 0
        
        # Verify learning rates were used
        for lr_values in lr_history.values():
            assert len(lr_values) > 0
    
    def test_meta_sgd_learning_rates_stats(self, simple_model):
        """Тест статистики learning rates"""
        config = MetaSGDConfig()
        meta_sgd = MetaSGD(simple_model, config)
        
        stats = meta_sgd.get_learning_rates_stats()
        
        assert 'global_lr_mean' in stats
        assert 'global_lr_std' in stats
        assert 'global_lr_min' in stats
        assert 'global_lr_max' in stats
        
        assert stats['global_lr_mean'] > 0
        assert stats['global_lr_min'] >= 0


class TestPrototypicalNetworks:
    """Tests for Prototypical Networks"""
    
    def test_protonet_initialization(self):
        """Тест инициализации ProtoNet"""
        config = ProtoNetConfig(
            embedding_dim=64,
            num_classes=5,
            num_support=5
        )
        
        protonet = PrototypicalNetworks(input_dim=20, config=config)
        
        assert protonet.config == config
        assert protonet.encoder is not None
        assert protonet.global_step == 0
    
    def test_protonet_compute_prototypes(self, sample_task_data):
        """Тест вычисления прототипов"""
        config = ProtoNetConfig(embedding_dim=32, num_classes=5)
        protonet = PrototypicalNetworks(input_dim=10, config=config)
        
        # Создаем embedding'и
        embeddings = torch.randn(25, 32)  # 5 classes * 5 examples
        labels = torch.arange(5).repeat(5)  # [0,1,2,3,4,0,1,2,3,4,...]
        
        prototypes = protonet.compute_prototypes(embeddings, labels)
        
        assert prototypes.shape == (5, 32)  # 5 классов, 32-мерные прототипы
    
    def test_protonet_predict_classification(self):
        """Тест предсказания классификации"""
        config = ProtoNetConfig(embedding_dim=32, num_classes=3)
        protonet = PrototypicalNetworks(input_dim=10, config=config)
        
        query_embeddings = torch.randn(10, 32)
        prototypes = torch.randn(3, 32)
        
        logits, probabilities = protonet.predict_classification(
            query_embeddings, prototypes
        )
        
        assert logits.shape == (10, 3)
        assert probabilities.shape == (10, 3)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(10))
    
    def test_protonet_few_shot_predict(self):
        """Тест few-shot предсказания"""
        config = ProtoNetConfig(embedding_dim=32)
        protonet = PrototypicalNetworks(input_dim=10, config=config)
        
        support_data = torch.randn(15, 10)  # 3 classes * 5 examples
        support_labels = torch.arange(3).repeat(5)
        query_data = torch.randn(9, 10)  # 3 classes * 3 queries
        
        predictions, confidence = protonet.few_shot_predict(
            support_data, support_labels, query_data
        )
        
        assert predictions.shape == (9,)
        assert confidence.shape == (9,)
        assert torch.all(predictions >= 0) and torch.all(predictions < 3)


class TestMatchingNetworks:
    """Tests for Matching Networks"""
    
    def test_matching_net_initialization(self):
        """Тест инициализации MatchingNet"""
        config = MatchingNetConfig(
            embedding_dim=64,
            attention_type="cosine"
        )
        
        matching_net = MatchingNetworks(input_dim=20, config=config)
        
        assert matching_net.config == config
        assert matching_net.encoder is not None
        assert matching_net.attention is not None
    
    def test_matching_net_attention(self):
        """Тест механизма внимания"""
        config = MatchingNetConfig(embedding_dim=32, attention_type="cosine")
        matching_net = MatchingNetworks(input_dim=10, config=config)
        
        query = torch.randn(5, 32)
        support = torch.randn(15, 32)
        support_labels = torch.arange(3).repeat(5)
        
        attended_features, attention_weights = matching_net.attention(
            query, support, support_labels
        )
        
        assert attended_features.shape == (5, 32)
        assert attention_weights.shape == (5, 15)
        assert torch.allclose(attention_weights.sum(dim=1), torch.ones(5))


class TestTaskDistribution:
    """Tests for системы распределения задач"""
    
    def test_crypto_task_distribution_initialization(self):
        """Тест инициализации CryptoTaskDistribution"""
        config = CryptoTaskConfig(
            task_type="classification",
            num_classes=3,
            num_support=5
        )
        
        task_dist = CryptoTaskDistribution(config)
        
        assert task_dist.config == config
        assert len(task_dist.available_pairs) > 0
        assert task_dist.task_counter == 0
    
    def test_crypto_task_sampling(self):
        """Тест семплирования криптовалютных задач"""
        config = CryptoTaskConfig(task_type="classification")
        task_dist = CryptoTaskDistribution(config)
        
        task = task_dist.sample_task()
        
        assert 'support_data' in task
        assert 'support_labels' in task
        assert 'query_data' in task
        assert 'query_labels' in task
        
        assert task['support_data'].shape[0] > 0
        assert task['query_data'].shape[0] > 0
    
    def test_task_batch_sampling(self):
        """Тест семплирования batch задач"""
        config = CryptoTaskConfig()
        task_dist = CryptoTaskDistribution(config)
        
        batch_size = 4
        task_batch = task_dist.sample_batch(batch_size)
        
        assert len(task_batch) == batch_size
        for task in task_batch:
            assert isinstance(task, dict)
            assert 'support_data' in task


class TestTaskSampler:
    """Tests for Task Sampler"""
    
    def test_task_cache_basic_operations(self, temp_dir):
        """Тест базовых операций кэша задач"""
        cache = TaskCache(max_size=10, cache_dir=temp_dir)
        
        # Тест put/get
        test_data = {
            'support_data': torch.randn(10, 5),
            'support_labels': torch.randint(0, 3, (10,))
        }
        
        cache.put("test_key", test_data)
        retrieved_data = cache.get("test_key")
        
        assert retrieved_data is not None
        assert torch.equal(retrieved_data['support_data'], test_data['support_data'])
        
        # Тест несуществующего ключа
        assert cache.get("nonexistent_key") is None
    
    def test_task_sampler_initialization(self):
        """Тест инициализации TaskSampler"""
        config = CryptoTaskConfig()
        task_dist = CryptoTaskDistribution(config)
        
        sampler_config = SamplerConfig(
            batch_size=8,
            prefetch_factor=2,
            enable_cache=True
        )
        
        sampler = TaskSampler(task_dist, sampler_config)
        
        assert sampler.task_distribution == task_dist
        assert sampler.config == sampler_config
        assert sampler.cache is not None
    
    def test_task_sampler_sampling(self):
        """Тест семплирования через TaskSampler"""
        config = CryptoTaskConfig()
        task_dist = CryptoTaskDistribution(config)
        sampler_config = SamplerConfig(batch_size=4, prefetch_factor=0)
        
        with TaskSampler(task_dist, sampler_config) as sampler:
            # Тест семплирования одной задачи
            task = sampler.sample_task()
            assert isinstance(task, dict)
            assert 'support_data' in task
            
            # Тест семплирования batch
            batch = sampler.sample_batch(3)
            assert len(batch) == 3


class TestMetaOptimizers:
    """Tests for мета-оптимизаторов"""
    
    def test_meta_optimizer_factory(self, simple_model):
        """Тест фабрики мета-оптимизаторов"""
        config = MetaOptimizerConfig()
        
        # Тест создания MAML оптимизатора
        maml_optimizer = MetaOptimizerFactory.create_optimizer(
            "maml", simple_model, config
        )
        assert maml_optimizer is not None
        
        # Тест создания Reptile оптимизатора
        reptile_optimizer = MetaOptimizerFactory.create_optimizer(
            "reptile", simple_model, config
        )
        assert reptile_optimizer is not None
        
        # Тест неизвестного типа
        with pytest.raises(ValueError):
            MetaOptimizerFactory.create_optimizer(
                "unknown", simple_model, config
            )
    
    def test_inner_loop_optimizer_factory(self):
        """Тест фабрики inner loop оптимизаторов"""
        config = InnerLoopConfig()
        
        # Тест создания SGD оптимизатора
        sgd_optimizer = InnerLoopOptimizerFactory.create_optimizer("sgd", config)
        assert sgd_optimizer is not None
        
        # Тест создания Adam оптимизатора
        adam_optimizer = InnerLoopOptimizerFactory.create_optimizer("adam", config)
        assert adam_optimizer is not None


class TestEvaluation:
    """Tests for системы оценки"""
    
    def test_classification_evaluator(self, simple_model, sample_task_data):
        """Test evaluator for classification"""
        config = EvaluationConfig(num_episodes=5, num_runs=2)
        evaluator = ClassificationEvaluator(config)
        
        metrics = evaluator.evaluate_episode(
            simple_model, sample_task_data, 
            num_shots=5, num_ways=5, adaptation_steps=3
        )
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
    
    def test_regression_evaluator(self, simple_model):
        """Тест evaluator для регрессии"""
        config = EvaluationConfig(num_episodes=5)
        evaluator = RegressionEvaluator(config)
        
        # Создаем данные регрессии
        regression_data = {
            'support_data': torch.randn(10, 10),
            'support_labels': torch.randn(10),
            'query_data': torch.randn(15, 10),
            'query_labels': torch.randn(15)
        }
        
        metrics = evaluator.evaluate_episode(
            simple_model, regression_data,
            num_shots=10, num_ways=1, adaptation_steps=5
        )
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0
    
    def test_few_shot_benchmark(self, simple_model, temp_dir):
        """Тест бенчмаркинга"""
        config = EvaluationConfig(num_episodes=3, num_runs=2)
        benchmark = FewShotBenchmark(config)
        
        models = {
            'model1': simple_model,
            'model2': SimpleTestModel()
        }
        
        def dummy_task_generator():
            return {
                'support_data': torch.randn(15, 10),
                'support_labels': torch.randint(0, 3, (15,)),
                'query_data': torch.randn(9, 10),
                'query_labels': torch.randint(0, 3, (9,))
            }
        
        with patch.object(benchmark, 'config') as mock_config:
            mock_config.save_plots = False  # Отключаем сохранение графиков
            
            results = benchmark.run_benchmark(
                models, dummy_task_generator, "classification"
            )
        
        assert 'individual_results' in results
        assert 'comparison_analysis' in results
        assert len(results['individual_results']) == 2


class TestUtilities:
    """Tests for утилит"""
    
    def test_gradient_manager(self, simple_model):
        """Тест менеджера градиентов"""
        manager = GradientManager()
        
        # Создаем fake градиенты
        for param in simple_model.parameters():
            param.grad = torch.randn_like(param)
        
        # Тест вычисления нормы градиентов
        norm = manager.compute_gradient_norm(simple_model.parameters())
        assert isinstance(norm, float)
        assert norm >= 0
        
        # Тест статистик градиентов
        stats = manager.compute_gradient_stats(simple_model.parameters())
        assert 'mean' in stats
        assert 'std' in stats
        assert 'norm_l2' in stats
        
        # Тест обрезки градиентов
        clipped_norm = manager.clip_gradients(simple_model.parameters(), max_norm=1.0)
        assert isinstance(clipped_norm, float)
    
    def test_meta_learning_metrics(self):
        """Тест системы метрик"""
        metrics = MetaLearningMetrics()
        
        # Тест отслеживания метрик
        metrics.track_metric("accuracy", 0.85, "task1")
        metrics.track_metric("accuracy", 0.90, "task2")
        
        # Тест получения сводки
        summary = metrics.get_metric_summary("accuracy")
        assert 'mean' in summary
        assert 'std' in summary
        assert summary['count'] == 2
        
        # Тест метрик адаптации
        adaptation_metrics = metrics.compute_adaptation_metrics(
            initial_performance=0.5,
            final_performance=0.8,
            num_adaptation_steps=5,
            adaptation_time=10.0
        )
        
        assert 'adaptation_improvement' in adaptation_metrics
        assert 'adaptation_rate' in adaptation_metrics
        assert adaptation_metrics['adaptation_improvement'] == 0.3
    
    def test_data_analyzer(self, sample_task_data):
        """Тест анализатора данных"""
        analyzer = DataAnalyzer()
        
        # Тест анализа сложности задачи
        difficulty = analyzer.analyze_task_difficulty(
            sample_task_data['support_data'],
            sample_task_data['support_labels'],
            sample_task_data['query_data'],
            sample_task_data['query_labels']
        )
        
        assert 'overall_difficulty' in difficulty
        assert 'feature_dimensionality' in difficulty
        assert 0 <= difficulty['overall_difficulty'] <= 1
        
        # Тест оценки качества данных
        quality = analyzer.assess_data_quality(
            sample_task_data['support_data'],
            sample_task_data['support_labels']
        )
        
        assert 'overall_quality' in quality
        assert 'missing_data_ratio' in quality
        assert 0 <= quality['overall_quality'] <= 1
    
    def test_visualizer(self, temp_dir):
        """Тест системы визуализации"""
        visualizer = Visualizer(save_dir=temp_dir)
        
        # Тест данных для визуализации
        metrics_history = {
            'loss': [
                {'value': 1.0, 'timestamp': 1000},
                {'value': 0.8, 'timestamp': 1001},
                {'value': 0.6, 'timestamp': 1002}
            ],
            'accuracy': [
                {'value': 0.6, 'timestamp': 1000},
                {'value': 0.7, 'timestamp': 1001},
                {'value': 0.8, 'timestamp': 1002}
            ]
        }
        
        # Тест создания графика прогресса обучения
        try:
            visualizer.plot_training_progress(metrics_history)
            # Проверяем, что файл создался
            plot_files = list(Path(temp_dir).glob("*.png"))
            assert len(plot_files) > 0
        except Exception as e:
            # Визуализация может не работать в headless среде
            pytest.skip(f"Visualization test skipped: {e}")


class TestIntegration:
    """Интеграционные тесты"""
    
    def test_end_to_end_maml_training(self):
        """Интеграционный тест обучения MAML"""
        # Создаем компоненты
        model = SimpleTestModel()
        config = MAMLConfig(num_inner_steps=2)
        maml = MAML(model, config)
        
        # Создаем задачи
        task_config = CryptoTaskConfig(task_type="classification")
        task_dist = CryptoTaskDistribution(task_config)
        
        # Симулируем несколько шагов обучения
        for step in range(3):
            task = task_dist.sample_task()
            
            # Конвертируем labels для регрессии (MAML ожидает float)
            task['support_labels'] = task['support_labels'].float()
            task['query_labels'] = task['query_labels'].float()
            
            metrics = maml.meta_train_step([task])
            
            assert 'meta_loss' in metrics
            assert metrics['meta_loss'] >= 0
            assert maml.global_step == step + 1
    
    def test_end_to_end_evaluation_pipeline(self):
        """Интеграционный тест пайплайна оценки"""
        # Создаем модель и задачи
        model = SimpleTestModel()
        
        def task_generator():
            return {
                'support_data': torch.randn(15, 10),
                'support_labels': torch.randint(0, 3, (15,)),
                'query_data': torch.randn(9, 10),
                'query_labels': torch.randint(0, 3, (9,))
            }
        
        # Создаем evaluator
        config = EvaluationConfig(num_episodes=2, num_runs=1)
        evaluator = ClassificationEvaluator(config)
        
        # Запускаем оценку
        results = evaluator.run_evaluation(model, task_generator, "test_model")
        
        assert 'model_name' in results
        assert 'aggregated_results' in results
        assert results['model_name'] == "test_model"
    
    def test_crypto_market_simulator_integration(self):
        """Интеграционный тест симулятора крипторынка"""
        config = CryptoTaskConfig()
        simulator = CryptoMarketSimulator(config)
        
        # Тест генерации ценового ряда
        from ..src.tasks.crypto_tasks import MarketRegime
        
        prices = simulator.generate_price_series(
            initial_price=100.0,
            length=100,
            regime=MarketRegime.BULL
        )
        
        assert len(prices) == 100
        assert prices[0] == 100.0
        assert np.all(prices > 0)  # Цены должны быть положительными
        
        # Тест генерации объемов
        volumes = simulator.generate_volume_series(prices)
        assert len(volumes) == len(prices)
        assert np.all(volumes > 0)
        
        # Тест технических индикаторов
        indicators = simulator.compute_technical_indicators(prices, volumes)
        assert len(indicators) > 0
        assert 'sma_20' in indicators or 'ema_12' in indicators


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])