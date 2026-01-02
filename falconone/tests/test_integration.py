"""
Integration tests for FalconOne
Tests multi-component workflows and interactions
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import time


@pytest.mark.integration
class TestAIMLPipeline:
    """Integration tests for AI/ML pipeline"""
    
    def test_online_learning_with_explainability(self):
        """Test online learning + explainable AI integration"""
        from falconone.ai.online_learning import OnlineLearner
        from falconone.ai.explainable_ai import ExplainableAI
        
        # Create online learner
        learner = OnlineLearner(logger=Mock())
        learner.initialize_model(n_features=8, n_classes=3)
        learner.warmup_samples = 20
        
        # Train with data
        X_train = np.random.rand(100, 8)
        y_train = np.random.randint(0, 3, 100)
        learner.partial_fit(X_train, y_train)
        
        # Create explainer
        explainer = ExplainableAI(logger=Mock())
        
        # Get explanation for prediction
        try:
            explanation = explainer.explain_prediction(
                model=learner.model,
                instance=X_train[0],
                method='shap',
                background_data=X_train[:30]
            )
            
            assert explanation is not None
            assert len(explanation.feature_importance) == 8
        except ImportError:
            pytest.skip("SHAP not available")
    
    def test_federated_learning_with_drift_detection(self):
        """Test federated learning with online learning drift detection"""
        from falconone.ai.federated_coordinator import FederatedCoordinator
        from falconone.ai.online_learning import OnlineLearner
        
        # Create federated coordinator
        config = {
            'num_clients': 5,
            'local_epochs': 3,
            'learning_rate': 0.01,
            'enable_byzantine_robust': True,
            'byzantine_method': 'krum'
        }
        coordinator = FederatedCoordinator(config, logger=Mock())
        
        # Create online learner for local updates
        learner = OnlineLearner(logger=Mock())
        learner.initialize_model(n_features=10, n_classes=2)
        
        # Simulate federated round with drift detection
        client_weights = []
        for i in range(config['num_clients']):
            X_local = np.random.rand(50, 10)
            y_local = np.random.randint(0, 2, 50)
            
            learner.partial_fit(X_local, y_local)
            
            # Extract weights (simplified)
            weights = {
                'layer_0': np.random.rand(10, 5),
                'layer_1': np.random.rand(5, 2)
            }
            client_weights.append(weights)
        
        # Aggregate with Byzantine robustness
        global_weights = coordinator.byzantine_robust_aggregation(
            client_weights,
            method='krum'
        )
        
        assert global_weights is not None
        assert 'layer_0' in global_weights


@pytest.mark.integration
class TestORANPipeline:
    """Integration tests for O-RAN components"""
    
    @patch('socket.socket')
    def test_ric_xapp_deployment(self, mock_socket):
        """Test RIC platform + xApp deployment flow"""
        from falconone.oran.near_rt_ric import NearRTRIC
        from falconone.oran.ric_xapp import TrafficSteeringXApp
        
        # Create RIC
        ric = NearRTRIC(e2_endpoint='localhost:36421', logger=Mock())
        
        # Mock socket connection
        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock
        
        ric.start()
        
        # Deploy xApp
        xapp_config = {
            'steering_threshold': 0.7,
            'iteration_interval': 1.0
        }
        
        success = ric.deploy_xapp(
            xapp_class=TrafficSteeringXApp,
            name='traffic_steering',
            config=xapp_config,
            service_models=['E2SM-KPM', 'E2SM-RC']
        )
        
        assert success == True
        assert 'traffic_steering' in ric.xapps
        
        # Stop xApp
        ric.stop_xapp('traffic_steering')
        
        xapp_desc = ric.xapps.get('traffic_steering')
        if xapp_desc:
            assert xapp_desc.state.name in ['STOPPED', 'STOPPING']
        
        ric.stop()
    
    def test_e2_subscription_to_xapp_flow(self, sample_kpi_report):
        """Test E2 subscription -> indication -> xApp handling"""
        from falconone.oran.near_rt_ric import NearRTRIC
        from falconone.oran.ric_xapp import AnomalyDetectionXApp
        
        # Create RIC
        ric = NearRTRIC(e2_endpoint='localhost:36421', logger=Mock())
        ric.e2_interface.connected = True  # Mock connection
        
        # Deploy anomaly detection xApp
        xapp_config = {
            'anomaly_threshold': 2.0,
            'iteration_interval': 1.0
        }
        
        ric.deploy_xapp(
            xapp_class=AnomalyDetectionXApp,
            name='anomaly_detector',
            config=xapp_config,
            service_models=['E2SM-KPM']
        )
        
        # Register xApp for RAN function 2 (KPM)
        ric.register_indication_router(
            ran_function_id=2,
            xapp_name='anomaly_detector'
        )
        
        # Simulate E2 indication
        from falconone.oran.e2_interface import E2Indication
        indication = E2Indication(
            request_id=1,
            ran_function_id=2,
            action_id=1,
            indication_type='report',
            indication_header=b'header',
            indication_message=b'kpi_data'
        )
        
        # Route indication to xApp
        ric.route_indication(indication)
        
        assert ric.statistics['indications_routed'] == 1
        
        ric.stop()


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations"""
    
    def test_complete_target_workflow(self, temp_db):
        """Test complete target tracking workflow"""
        from falconone.utils.database import FalconOneDatabase
        
        db = FalconOneDatabase(db_path=temp_db, logger=Mock())
        
        # Create target
        target_id = db.create_target(
            imsi='001010000000001',
            imei='123456789012345',
            msisdn='+1234567890'
        )
        
        assert target_id is not None
        
        # Get target
        target = db.get_target(target_id)
        assert target['imsi'] == '001010000000001'
        
        # Update target
        db.update_target(
            target_id,
            location_lat=37.7749,
            location_lon=-122.4194
        )
        
        # Verify update
        updated_target = db.get_target(target_id)
        assert updated_target['location_lat'] == 37.7749
        
        # List targets
        targets = db.list_targets()
        assert len(targets) == 1
        
        # Delete target
        db.delete_target(target_id)
        
        # Verify deletion
        deleted_target = db.get_target(target_id)
        assert deleted_target is None or deleted_target.get('is_active') == 0
    
    def test_suci_capture_with_target_link(self, temp_db):
        """Test SUCI capture with foreign key to target"""
        from falconone.utils.database import FalconOneDatabase
        import time
        
        db = FalconOneDatabase(db_path=temp_db, logger=Mock())
        
        # Create target
        target_id = db.create_target(
            imsi='001010000000001',
            imei='123456789012345'
        )
        
        # Store SUCI capture
        suci_id = db.store_suci_capture(
            target_id=target_id,
            suci='suci-0-001-01-0-0-12345678',
            capture_time=time.time(),
            network_id='00101',
            cell_id=12345
        )
        
        assert suci_id is not None
        
        # Retrieve SUCI capture
        suci_capture = db.get_suci_capture(suci_id)
        assert suci_capture['target_id'] == target_id
        assert suci_capture['suci'] == 'suci-0-001-01-0-0-12345678'


@pytest.mark.integration
class TestExploitPipeline:
    """Integration tests for exploit workflows"""
    
    @patch('falconone.sdr.manager.SDRDevice')
    def test_dos_attack_with_sdr(self, mock_sdr_device):
        """Test DOS attack with SDR integration"""
        from falconone.exploit.exploit_engine import ExploitEngine
        
        # Mock SDR
        mock_sdr = Mock()
        mock_sdr.sample_rate = 2.4e6
        mock_sdr.center_freq = 900e6
        mock_sdr.write_samples = Mock()
        mock_sdr_device.return_value = mock_sdr
        
        # Create exploit engine
        engine = ExploitEngine(logger=Mock())
        
        # Execute DOS attack
        result = engine.execute_dos_attack(
            target_frequency=900e6,
            attack_type='frequency_jamming',
            duration_sec=1,
            power_dbm=20
        )
        
        assert result['success'] == True
        assert result['attack_type'] == 'frequency_jamming'
    
    def test_mitm_attack_workflow(self):
        """Test MITM attack complete workflow"""
        from falconone.exploit.exploit_engine import ExploitEngine
        
        engine = ExploitEngine(logger=Mock())
        
        # Mock SDR initialization
        engine.sdr_manager = Mock()
        engine.sdr_manager.get_device = Mock(return_value=Mock())
        
        with patch.object(engine, '_broadcast_fake_cell') as mock_broadcast:
            mock_broadcast.return_value = True
            
            with patch.object(engine, '_wait_for_attachment') as mock_wait:
                mock_wait.return_value = {'imsi': '001010000000001', 'attached': True}
                
                result = engine.execute_mitm_attack(
                    target_network='00101',
                    target_frequency=900e6,
                    duration_sec=1
                )
                
                assert result['success'] == True
                assert 'captured_credentials' in result


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndScenarios:
    """End-to-end integration tests"""
    
    def test_signal_detection_to_classification_to_explanation(self):
        """Test: Signal detection -> Classification -> Explainability"""
        from falconone.monitoring.fiveg_monitor import FiveGMonitor
        from falconone.ai.signal_classifier import SignalClassifier
        from falconone.ai.explainable_ai import ExplainableAI
        
        # 1. Monitor signal (mock)
        monitor = FiveGMonitor(logger=Mock())
        signal_data = {
            'iq_data': np.random.randn(1000) + 1j * np.random.randn(1000),
            'center_frequency': 3.5e9,
            'bandwidth': 100e6,
            'timestamp': time.time()
        }
        
        # 2. Classify signal
        classifier = SignalClassifier(logger=Mock())
        
        # Mock model
        with patch.object(classifier, 'model') as mock_model:
            mock_model.predict.return_value = np.array([2])  # Class 2
            mock_model.predict_proba.return_value = np.array([[0.1, 0.2, 0.7]])
            
            classification = {
                'signal_type': 'PSS',
                'confidence': 0.7,
                'features': np.random.rand(10)
            }
            
            # 3. Explain classification
            explainer = ExplainableAI(logger=Mock())
            
            try:
                explanation = explainer.explain_prediction(
                    model=mock_model,
                    instance=classification['features'],
                    method='shap',
                    background_data=np.random.rand(30, 10)
                )
                
                assert explanation is not None
                assert explanation.prediction == 2
            except ImportError:
                pytest.skip("SHAP not available")
    
    def test_target_tracking_with_online_learning(self, temp_db):
        """Test: Target tracking -> Feature extraction -> Online learning"""
        from falconone.utils.database import FalconOneDatabase
        from falconone.ai.online_learning import OnlineLearner
        
        db = FalconOneDatabase(db_path=temp_db, logger=Mock())
        learner = OnlineLearner(logger=Mock())
        learner.initialize_model(n_features=5, n_classes=2)
        learner.warmup_samples = 10
        
        # Simulate multiple target observations
        for i in range(50):
            # Create/update target
            target_id = db.create_target(
                imsi=f'00101000000000{i:02d}',
                imei=f'12345678901234{i}'
            )
            
            # Extract features (mock)
            features = np.random.rand(5)
            label = np.random.randint(0, 2)
            
            # Update model
            learner.partial_fit(features.reshape(1, -1), np.array([label]))
        
        # Verify learning progress
        assert learner.num_updates > 0
        
        # Predict on new sample
        new_features = np.random.rand(1, 5)
        prediction = learner.predict(new_features)
        
        assert prediction is not None
        assert prediction[0] in [0, 1]
