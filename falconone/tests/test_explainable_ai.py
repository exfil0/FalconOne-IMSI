"""
Unit tests for Explainable AI module (SHAP & LIME)
Tests SHAP integration, LIME integration, hybrid mode, and visualizations
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os


@pytest.fixture
def explainable_ai():
    """Create ExplainableAI instance"""
    from falconone.ai.explainable_ai import ExplainableAI
    return ExplainableAI(logger=Mock())


@pytest.fixture
def simple_model():
    """Simple linear model for testing"""
    from sklearn.linear_model import LogisticRegression
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model = LogisticRegression()
    model.fit(X, y)
    return model, X


class TestExplainableAI:
    """Test suite for ExplainableAI class"""
    
    def test_initialization(self, explainable_ai):
        """Test ExplainableAI initialization"""
        assert explainable_ai is not None
        assert explainable_ai.explainer_cache == {}
        assert explainable_ai.statistics['total_explanations'] == 0
    
    @patch('falconone.ai.explainable_ai.shap')
    def test_explain_prediction_shap(self, mock_shap, explainable_ai, simple_model):
        """Test SHAP explanation generation"""
        model, X = simple_model
        instance = X[0]
        
        # Mock SHAP explainer
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.array([[0.1, -0.2, 0.3, -0.1, 0.05]])
        mock_explainer.expected_value = 0.5
        mock_shap.LinearExplainer.return_value = mock_explainer
        
        explanation = explainable_ai.explain_prediction(
            model=model,
            instance=instance,
            method='shap',
            background_data=X[:50]
        )
        
        assert explanation is not None
        assert explanation.explanation_type == 'shap'
        assert 'feature_importance' in explanation.__dict__
        assert len(explanation.top_features) <= 10
        assert 0.0 <= explanation.explanation_confidence <= 1.0
    
    @patch('falconone.ai.explainable_ai.lime')
    def test_explain_prediction_lime(self, mock_lime, explainable_ai, simple_model):
        """Test LIME explanation generation"""
        model, X = simple_model
        instance = X[0]
        
        # Mock LIME explainer
        mock_explainer = MagicMock()
        mock_exp = MagicMock()
        mock_exp.as_list.return_value = [(0, 0.1), (1, -0.2), (2, 0.3)]
        mock_exp.score = 0.85
        mock_explainer.explain_instance.return_value = mock_exp
        mock_lime.lime_tabular.LimeTabularExplainer.return_value = mock_explainer
        
        explanation = explainable_ai.explain_prediction(
            model=model,
            instance=instance,
            method='lime',
            background_data=X[:50]
        )
        
        assert explanation is not None
        assert explanation.explanation_type == 'lime'
        assert len(explanation.top_features) > 0
        assert explanation.explanation_confidence == 0.85
    
    @patch('falconone.ai.explainable_ai.shap')
    @patch('falconone.ai.explainable_ai.lime')
    def test_explain_prediction_hybrid(self, mock_lime, mock_shap, explainable_ai, simple_model):
        """Test hybrid SHAP+LIME explanation"""
        model, X = simple_model
        instance = X[0]
        
        # Mock both explainers
        mock_shap_explainer = MagicMock()
        mock_shap_explainer.shap_values.return_value = np.array([[0.1, -0.2, 0.3, -0.1, 0.05]])
        mock_shap_explainer.expected_value = 0.5
        mock_shap.LinearExplainer.return_value = mock_shap_explainer
        
        mock_lime_explainer = MagicMock()
        mock_exp = MagicMock()
        mock_exp.as_list.return_value = [(0, 0.15), (1, -0.18), (2, 0.25)]
        mock_exp.score = 0.85
        mock_lime_explainer.explain_instance.return_value = mock_exp
        mock_lime.lime_tabular.LimeTabularExplainer.return_value = mock_lime_explainer
        
        explanation = explainable_ai.explain_prediction(
            model=model,
            instance=instance,
            method='hybrid',
            background_data=X[:50]
        )
        
        assert explanation is not None
        assert explanation.explanation_type == 'hybrid'
        # Should combine both methods
        assert len(explanation.feature_importance) > 0
    
    def test_explain_batch(self, explainable_ai, simple_model):
        """Test batch explanation generation"""
        model, X = simple_model
        instances = X[:5]
        
        with patch.object(explainable_ai, '_explain_shap') as mock_shap:
            mock_shap.return_value = MagicMock(
                explanation_type='shap',
                feature_importance={'feature_0': 0.1},
                top_features=[('feature_0', 0.1)],
                explanation_confidence=0.9
            )
            
            explanations = explainable_ai.explain_batch(
                model=model,
                instances=instances,
                method='shap',
                background_data=X[:50]
            )
            
            assert len(explanations) == 5
            assert all(exp.explanation_type == 'shap' for exp in explanations)
    
    @patch('falconone.ai.explainable_ai.shap')
    def test_get_global_importance(self, mock_shap, explainable_ai, simple_model):
        """Test global feature importance extraction"""
        model, X = simple_model
        
        # Mock SHAP explainer
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.array([
            [0.1, -0.2, 0.3, -0.1, 0.05],
            [0.15, -0.18, 0.25, -0.12, 0.08],
        ])
        mock_shap.LinearExplainer.return_value = mock_explainer
        
        importance = explainable_ai.get_global_importance(
            model=model,
            background_data=X[:50]
        )
        
        assert isinstance(importance, dict)
        assert len(importance) == 5
        # Should be sorted by absolute value
        values = list(importance.values())
        assert values == sorted(values, reverse=True)
    
    def test_export_explanation_json(self, explainable_ai):
        """Test JSON export of explanation"""
        from falconone.ai.explainable_ai import Explanation
        
        explanation = Explanation(
            explanation_type='shap',
            feature_importance={'feature_0': 0.5, 'feature_1': -0.3},
            prediction=1,
            prediction_confidence=0.85,
            explanation_confidence=0.92,
            top_features=[('feature_0', 0.5), ('feature_1', -0.3)],
            visualization_path=None
        )
        
        result = explainable_ai.export_explanation(explanation, format='json')
        
        assert result is not None
        assert 'explanation_type' in result
        assert 'feature_importance' in result
        assert result['explanation_type'] == 'shap'
    
    def test_export_explanation_html(self, explainable_ai):
        """Test HTML export of explanation"""
        from falconone.ai.explainable_ai import Explanation
        
        explanation = Explanation(
            explanation_type='lime',
            feature_importance={'feature_0': 0.4, 'feature_1': 0.2},
            prediction=0,
            prediction_confidence=0.78,
            explanation_confidence=0.88,
            top_features=[('feature_0', 0.4), ('feature_1', 0.2)],
            visualization_path=None
        )
        
        html_output = explainable_ai.export_explanation(explanation, format='html')
        
        assert html_output is not None
        assert '<html>' in html_output
        assert 'feature_0' in html_output
    
    def test_statistics_tracking(self, explainable_ai, simple_model):
        """Test statistics tracking"""
        model, X = simple_model
        
        with patch.object(explainable_ai, '_explain_shap') as mock_shap:
            mock_shap.return_value = MagicMock(
                explanation_type='shap',
                feature_importance={},
                top_features=[],
                explanation_confidence=0.9
            )
            
            explainable_ai.explain_prediction(model, X[0], 'shap', X[:50])
            explainable_ai.explain_prediction(model, X[1], 'shap', X[:50])
            
            stats = explainable_ai.get_statistics()
            assert stats['total_explanations'] == 2
            assert stats['avg_explanation_time'] > 0
    
    def test_invalid_method(self, explainable_ai, simple_model):
        """Test invalid explanation method"""
        model, X = simple_model
        
        with pytest.raises(ValueError):
            explainable_ai.explain_prediction(
                model=model,
                instance=X[0],
                method='invalid_method',
                background_data=X[:50]
            )
    
    @patch('falconone.ai.explainable_ai.plt')
    def test_visualization_creation(self, mock_plt, explainable_ai, simple_model):
        """Test visualization file creation"""
        model, X = simple_model
        
        with patch.object(explainable_ai, '_explain_shap') as mock_shap:
            explanation = MagicMock()
            explanation.explanation_type = 'shap'
            explanation.feature_importance = {'f0': 0.5, 'f1': -0.3}
            explanation.visualization_path = '/tmp/test.png'
            mock_shap.return_value = explanation
            
            result = explainable_ai.explain_prediction(
                model=model,
                instance=X[0],
                method='shap',
                background_data=X[:50],
                visualize=True
            )
            
            assert result.visualization_path is not None
    
    def test_explainer_caching(self, explainable_ai, simple_model):
        """Test explainer instance caching"""
        model, X = simple_model
        
        with patch('falconone.ai.explainable_ai.shap') as mock_shap:
            mock_explainer = MagicMock()
            mock_shap.LinearExplainer.return_value = mock_explainer
            
            # First call should create explainer
            explainable_ai._get_shap_explainer(model, X[:50])
            assert mock_shap.LinearExplainer.call_count == 1
            
            # Second call should use cache
            explainable_ai._get_shap_explainer(model, X[:50])
            assert mock_shap.LinearExplainer.call_count == 1  # Not called again
    
    def test_feature_names_handling(self, explainable_ai, simple_model):
        """Test custom feature names"""
        model, X = simple_model
        feature_names = ['signal_strength', 'frequency', 'bandwidth', 'latency', 'jitter']
        
        with patch.object(explainable_ai, '_explain_shap') as mock_shap:
            explanation = MagicMock()
            explanation.explanation_type = 'shap'
            explanation.feature_importance = {feature_names[0]: 0.5}
            explanation.top_features = [(feature_names[0], 0.5)]
            mock_shap.return_value = explanation
            
            result = explainable_ai.explain_prediction(
                model=model,
                instance=X[0],
                method='shap',
                background_data=X[:50],
                feature_names=feature_names
            )
            
            assert feature_names[0] in result.feature_importance


class TestExplanationDataclass:
    """Test Explanation dataclass"""
    
    def test_explanation_creation(self):
        """Test creating Explanation instance"""
        from falconone.ai.explainable_ai import Explanation
        
        exp = Explanation(
            explanation_type='shap',
            feature_importance={'f1': 0.5},
            prediction=1,
            prediction_confidence=0.85,
            explanation_confidence=0.92,
            top_features=[('f1', 0.5)],
            visualization_path='/tmp/exp.png'
        )
        
        assert exp.explanation_type == 'shap'
        assert exp.prediction == 1
        assert exp.prediction_confidence == 0.85
    
    def test_explanation_dict_conversion(self):
        """Test converting Explanation to dict"""
        from falconone.ai.explainable_ai import Explanation
        from dataclasses import asdict
        
        exp = Explanation(
            explanation_type='lime',
            feature_importance={'f1': 0.3, 'f2': -0.2},
            prediction=0,
            prediction_confidence=0.78,
            explanation_confidence=0.88,
            top_features=[('f1', 0.3), ('f2', -0.2)],
            visualization_path=None
        )
        
        exp_dict = asdict(exp)
        assert 'explanation_type' in exp_dict
        assert exp_dict['prediction'] == 0


@pytest.mark.slow
class TestExplainableAIIntegration:
    """Integration tests with real ML models"""
    
    def test_random_forest_shap_integration(self, explainable_ai):
        """Test SHAP with RandomForest"""
        from sklearn.ensemble import RandomForestClassifier
        
        X = np.random.rand(100, 8)
        y = np.random.randint(0, 3, 100)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        try:
            explanation = explainable_ai.explain_prediction(
                model=model,
                instance=X[0],
                method='shap',
                background_data=X[:30]
            )
            
            assert explanation is not None
            assert len(explanation.feature_importance) == 8
        except ImportError:
            pytest.skip("SHAP not available")
    
    def test_logistic_regression_lime_integration(self, explainable_ai):
        """Test LIME with LogisticRegression"""
        from sklearn.linear_model import LogisticRegression
        
        X = np.random.rand(100, 6)
        y = np.random.randint(0, 2, 100)
        model = LogisticRegression()
        model.fit(X, y)
        
        try:
            explanation = explainable_ai.explain_prediction(
                model=model,
                instance=X[0],
                method='lime',
                background_data=X[:30]
            )
            
            assert explanation is not None
            assert explanation.explanation_type == 'lime'
        except ImportError:
            pytest.skip("LIME not available")
