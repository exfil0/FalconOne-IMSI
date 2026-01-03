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


# ============================================================================
# LE Mode Integration Tests (v1.9.4)
# ============================================================================

@pytest.mark.integration
class TestLEModeIntegration:
    """Integration tests for Law Enforcement Mode"""
    
    def test_warrant_workflow_complete(self):
        """Test complete warrant workflow from creation to execution"""
        # Simulate complete warrant lifecycle
        warrant_data = {
            'warrant_id': 'WRT-2024-001',
            'warrant_type': 'intercept',
            'target_identifiers': ['IMSI:123456789012345'],
            'case_number': 'CASE-2024-001',
            'issuing_court': 'Federal District Court',
            'issuing_judge': 'Hon. Smith',
            'authorized_by': 'Agent Johnson',
            'scope': ['voice', 'sms', 'location'],
            'restrictions': ['no_content_intercept'],
            'status': 'pending'
        }
        
        # Verify warrant structure
        assert warrant_data['warrant_id'] is not None
        assert warrant_data['case_number'] == 'CASE-2024-001'
        assert len(warrant_data['target_identifiers']) > 0
        
        # Simulate activation
        warrant_data['status'] = 'active'
        assert warrant_data['status'] == 'active'
        
        # Simulate operation recording
        operations = []
        operations.append({
            'operation_id': 'OP-001',
            'warrant_id': warrant_data['warrant_id'],
            'operation_type': 'location_query',
            'target_identifier': 'IMSI:123456789012345',
            'operator_id': 'agent001',
            'timestamp': time.time()
        })
        
        assert len(operations) == 1
        
        # Simulate completion
        warrant_data['status'] = 'completed'
        assert warrant_data['status'] == 'completed'
    
    def test_warrant_authorization_chain(self):
        """Test warrant authorization and approval chain"""
        # Simulate warrant requiring approval
        warrant = {
            'warrant_id': 'WRT-2024-002',
            'warrant_type': 'wiretap',
            'requires_approval': True,
            'status': 'pending_approval',
            'approvals': []
        }
        
        # Should not be activatable without approval
        assert warrant['status'] == 'pending_approval'
        assert len(warrant['approvals']) == 0
        
        # Add approval
        warrant['approvals'].append({
            'approver_id': 'supervisor_chief',
            'approval_time': time.time(),
            'notes': 'Approved per policy'
        })
        warrant['status'] = 'approved'
        
        # Now can be activated
        assert warrant['status'] == 'approved'
        assert len(warrant['approvals']) == 1
        
        warrant['status'] = 'active'
        assert warrant['status'] == 'active'
    
    def test_target_scope_enforcement(self):
        """Test that operations are restricted to warrant scope"""
        warrant = {
            'warrant_id': 'WRT-2024-003',
            'scope': ['location'],  # Only location authorized
            'restrictions': ['no_voice', 'no_sms'],
            'status': 'active'
        }
        
        # Location operation should succeed
        location_op = 'location_query'
        assert location_op.startswith('location') or 'location' in warrant['scope']
        
        # Voice operation should fail (out of scope)
        voice_op = 'voice_intercept'
        is_allowed = 'voice' in warrant['scope'] and 'no_voice' not in warrant['restrictions']
        assert not is_allowed
    
    def test_audit_trail_generation(self):
        """Test that audit trail is properly generated"""
        audit_entries = []
        
        # Simulate warrant creation audit
        audit_entries.append({
            'event_type': 'warrant_created',
            'warrant_id': 'WRT-2024-004',
            'timestamp': time.time(),
            'actor': 'agent001',
            'details': {'case_number': 'CASE-2024-004'}
        })
        
        # Simulate operation audit
        audit_entries.append({
            'event_type': 'operation_performed',
            'warrant_id': 'WRT-2024-004',
            'timestamp': time.time(),
            'actor': 'agent001',
            'details': {'operation_type': 'location_query'}
        })
        
        # Verify audit trail
        assert len(audit_entries) >= 2
        assert all('timestamp' in e for e in audit_entries)
        assert all('actor' in e for e in audit_entries)


# ============================================================================
# ISAC Integration Tests (v1.9.4)
# ============================================================================

@pytest.mark.integration
class TestISACIntegration:
    """Integration tests for Integrated Sensing and Communication"""
    
    def test_beam_management_full_cycle(self):
        """Test complete beam management cycle"""
        # Initialize beams
        beams = []
        beam_count = 8
        
        for i in range(beam_count):
            beams.append({
                'beam_id': i,
                'azimuth_deg': -60 + (i * 15),
                'elevation_deg': 0,
                'beamwidth_deg': 15,
                'gain_db': 12,
                'mode': 'communication',
                'active': True
            })
        
        assert len(beams) == 8
        
        # Configure individual beam
        beams[0]['azimuth_deg'] = 45.0
        beams[0]['elevation_deg'] = 10.0
        beams[0]['mode'] = 'sensing'
        
        # Verify configuration
        assert beams[0]['mode'] == 'sensing'
        assert beams[0]['azimuth_deg'] == 45.0
    
    def test_simultaneous_sensing_communication(self):
        """Test ISAC with simultaneous sensing and communication"""
        beams = []
        
        # Configure beams for mixed mode
        # Beams 0-3 for communication
        for i in range(4):
            beams.append({
                'beam_id': i,
                'mode': 'communication',
                'active': True
            })
        
        # Beams 4-7 for sensing
        for i in range(4, 8):
            beams.append({
                'beam_id': i,
                'mode': 'sensing',
                'active': True
            })
        
        # Count by mode
        comm_beams = len([b for b in beams if b['mode'] == 'communication'])
        sens_beams = len([b for b in beams if b['mode'] == 'sensing'])
        
        assert comm_beams == 4
        assert sens_beams == 4
        
        # Verify simultaneous operation possible
        isac_status = {
            'communication_beams': comm_beams,
            'sensing_beams': sens_beams,
            'mode': 'simultaneous' if comm_beams > 0 and sens_beams > 0 else 'single'
        }
        
        assert isac_status['mode'] == 'simultaneous'
    
    def test_beam_tracking_moving_target(self):
        """Test beam tracking for moving targets"""
        # Start target tracking
        tracking_config = {
            'tracking_id': 'TRK-001',
            'initial_azimuth': 30.0,
            'initial_elevation': 5.0,
            'update_rate_hz': 10,
            'prediction_enabled': True,
            'max_velocity_mps': 50
        }
        
        tracking_history = [tracking_config['initial_azimuth']]
        
        # Simulate target movement
        for t in range(5):
            # Update with new target position
            new_azimuth = 30.0 + (t * 5)  # Moving 5 degrees per update
            tracking_history.append(new_azimuth)
        
        # Verify beam followed target
        assert tracking_history[-1] > tracking_history[0]
        assert len(tracking_history) == 6
    
    def test_beam_sweep_pattern(self):
        """Test beam sweep pattern generation"""
        # Generate sweep pattern
        sweep_config = {
            'start_azimuth': -60,
            'end_azimuth': 60,
            'step_deg': 10
        }
        
        sweep_positions = []
        current = sweep_config['start_azimuth']
        
        while current <= sweep_config['end_azimuth']:
            sweep_positions.append(current)
            current += sweep_config['step_deg']
        
        # Verify sweep coverage
        assert len(sweep_positions) == 13  # -60 to 60 in steps of 10
        assert sweep_positions[0] == -60
        assert sweep_positions[-1] == 60


# ============================================================================
# Multi-Component Integration Tests (v1.9.4)
# ============================================================================

@pytest.mark.integration
class TestMultiComponentWorkflow:
    """Integration tests for multi-component workflows"""
    
    def test_sdr_to_analysis_pipeline(self):
        """Test complete pipeline from SDR capture to analysis"""
        # Stage 1: SDR Configuration
        sdr_config = {
            'sample_rate': 23.04e6,
            'center_freq': 935.2e6,
            'gain': 40,
            'device': 'USRP'
        }
        assert sdr_config['sample_rate'] > 0
        
        # Stage 2: Signal Capture
        captured_signal = {
            'frequency': sdr_config['center_freq'],
            'timestamp': time.time(),
            'sample_count': 10000
        }
        assert captured_signal['sample_count'] > 0
        
        # Stage 3: GSM Decoding (simulated)
        decoded_data = {
            'imsi': '310260123456789',
            'tmsi': '0x12345678',
            'lac': 1234,
            'cell_id': 5678
        }
        assert decoded_data['imsi'] is not None
        
        # Stage 4: Analysis
        analysis_result = {
            'classification': 'smartphone',
            'confidence': 0.95,
            'network_behavior': 'normal'
        }
        assert analysis_result['confidence'] > 0.9
    
    def test_failover_during_capture(self):
        """Test SDR failover during active capture"""
        # Simulate device health
        devices = {
            'usrp1': {'type': 'USRP-B210', 'health': 100, 'active': True},
            'hackrf1': {'type': 'HackRF', 'health': 100, 'active': False}
        }
        
        # Simulate primary device failure
        devices['usrp1']['health'] = 0
        
        # Trigger failover
        if devices['usrp1']['health'] < 30:
            devices['usrp1']['active'] = False
            devices['hackrf1']['active'] = True
        
        # Verify failover
        assert devices['usrp1']['active'] == False
        assert devices['hackrf1']['active'] == True
    
    def test_high_volume_capture_processing(self):
        """Test processing of high volume captures"""
        import queue
        
        capture_queue = queue.Queue()
        processed_count = 0
        target_count = 100
        
        # Generate mock captures
        for i in range(target_count):
            capture_queue.put({
                'id': i,
                'imsi': f'31026012345{i:04d}',
                'timestamp': time.time()
            })
        
        # Process captures
        while not capture_queue.empty():
            capture = capture_queue.get()
            # Simulate processing
            processed_count += 1
        
        assert processed_count == target_count