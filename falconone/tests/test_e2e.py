"""
End-to-End tests for FalconOne
Tests complete user workflows from start to finish
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os
import time


@pytest.mark.e2e
class TestCompleteWorkflows:
    """End-to-end tests for complete user workflows"""
    
    def test_complete_target_tracking_workflow(self, temp_db):
        """E2E: Create target -> Scan -> Capture SUCI -> Analyze -> Export"""
        from falconone.utils.database import FalconOneDatabase
        from falconone.monitoring.fiveg_monitor import FiveGMonitor
        
        db = FalconOneDatabase(db_path=temp_db, logger=Mock())
        monitor = FiveGMonitor(logger=Mock())
        
        # Step 1: Create target
        target_id = db.create_target(
            imsi='001010000000001',
            imei='123456789012345',
            msisdn='+1234567890',
            metadata={'device_type': 'smartphone', 'os': 'Android'}
        )
        assert target_id is not None
        
        # Step 2: Scan for target (mock)
        scan_results = {
            'frequency': 3.5e9,
            'cell_id': 12345,
            'signals_detected': 5
        }
        
        # Step 3: Capture SUCI
        suci_id = db.store_suci_capture(
            target_id=target_id,
            suci='suci-0-001-01-0-0-12345678',
            capture_time=time.time(),
            network_id='00101',
            cell_id=12345
        )
        assert suci_id is not None
        
        # Step 4: Update target location
        db.update_target(
            target_id,
            location_lat=37.7749,
            location_lon=-122.4194,
            last_seen=time.time()
        )
        
        # Step 5: Retrieve all data
        target = db.get_target(target_id)
        assert target['imsi'] == '001010000000001'
        assert target['location_lat'] == 37.7749
        
        suci_captures = db.get_suci_captures_for_target(target_id)
        assert len(suci_captures) >= 1
        
        # Step 6: Export (mock)
        export_data = {
            'target': target,
            'suci_captures': suci_captures,
            'scan_results': scan_results
        }
        
        assert export_data['target']['imsi'] == '001010000000001'
    
    @patch('falconone.sdr.manager.SDRDevice')
    def test_complete_exploit_workflow(self, mock_sdr_device, temp_db):
        """E2E: Initialize SDR -> Select target -> Execute exploit -> Store results"""
        from falconone.exploit.exploit_engine import ExploitEngine
        from falconone.utils.database import FalconOneDatabase
        
        # Mock SDR
        mock_sdr = Mock()
        mock_sdr.sample_rate = 2.4e6
        mock_sdr.center_freq = 900e6
        mock_sdr.write_samples = Mock()
        mock_sdr_device.return_value = mock_sdr
        
        db = FalconOneDatabase(db_path=temp_db, logger=Mock())
        engine = ExploitEngine(logger=Mock())
        
        # Step 1: Create target
        target_id = db.create_target(
            imsi='001010000000002',
            imei='987654321098765'
        )
        
        # Step 2: Execute exploit
        result = engine.execute_dos_attack(
            target_frequency=900e6,
            attack_type='frequency_jamming',
            duration_sec=1,
            power_dbm=20
        )
        
        assert result['success'] == True
        
        # Step 3: Store exploit operation
        operation_id = db.store_exploit_operation(
            target_id=target_id,
            exploit_type='dos_attack',
            parameters={'frequency': 900e6, 'duration': 1},
            result=result,
            timestamp=time.time()
        )
        
        assert operation_id is not None
        
        # Step 4: Retrieve operation history
        operations = db.get_exploit_operations_for_target(target_id)
        assert len(operations) >= 1
        assert operations[0]['exploit_type'] == 'dos_attack'
    
    def test_complete_ml_pipeline_workflow(self, temp_db):
        """E2E: Train model -> Classify signals -> Get explanations -> Update online"""
        from falconone.ai.signal_classifier import SignalClassifier
        from falconone.ai.online_learning import OnlineLearner
        from falconone.ai.explainable_ai import ExplainableAI
        
        # Step 1: Initialize online learner
        learner = OnlineLearner(logger=Mock())
        learner.initialize_model(n_features=10, n_classes=3)
        learner.warmup_samples = 20
        
        # Step 2: Train with initial data
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 3, 100)
        learner.partial_fit(X_train, y_train)
        
        # Step 3: Classify new signal
        X_new = np.random.rand(1, 10)
        prediction = learner.predict(X_new)
        proba = learner.predict_proba(X_new)
        
        assert prediction is not None
        assert proba.shape == (1, 3)
        
        # Step 4: Get explanation
        explainer = ExplainableAI(logger=Mock())
        
        try:
            explanation = explainer.explain_prediction(
                model=learner.model,
                instance=X_new[0],
                method='shap',
                background_data=X_train[:30]
            )
            
            assert explanation is not None
            assert len(explanation.feature_importance) == 10
        except ImportError:
            pytest.skip("SHAP not available")
        
        # Step 5: Continue learning with new data
        X_new_batch = np.random.rand(20, 10)
        y_new_batch = np.random.randint(0, 3, 20)
        metrics = learner.partial_fit(X_new_batch, y_new_batch)
        
        assert metrics['num_updates'] >= 2
    
    @patch('socket.socket')
    def test_complete_oran_workflow(self, mock_socket, temp_db):
        """E2E: Start RIC -> Deploy xApp -> Connect E2 node -> Process KPIs"""
        from falconone.oran.near_rt_ric import NearRTRIC
        from falconone.oran.ric_xapp import AnomalyDetectionXApp
        from falconone.oran.e2_interface import E2Indication
        
        # Mock socket
        mock_sock = MagicMock()
        mock_socket.return_value = mock_sock
        
        # Step 1: Start RIC platform
        ric = NearRTRIC(e2_endpoint='localhost:36421', logger=Mock())
        ric.start()
        
        assert ric.running == True
        
        # Step 2: Deploy anomaly detection xApp
        xapp_config = {
            'anomaly_threshold': 2.0,
            'iteration_interval': 1.0
        }
        
        success = ric.deploy_xapp(
            xapp_class=AnomalyDetectionXApp,
            name='anomaly_detector',
            config=xapp_config,
            service_models=['E2SM-KPM']
        )
        
        assert success == True
        
        # Step 3: Register E2 node
        ric.e2_interface.connected = True
        ric.e2_interface.register_e2_node(
            node_id='test_gnb_001',
            ran_function_id=1,
            service_models=['E2SM-KPM', 'E2SM-RC'],
            plmn_id='00101',
            gnb_id='001',
            cell_ids=[1, 2, 3]
        )
        
        # Step 4: Subscribe to KPIs
        request_id = ric.e2_interface.subscribe(
            ran_function_id=2,
            service_model='E2SM-KPM',
            event_triggers={},
            actions=[],
            report_period_ms=1000
        )
        
        assert request_id is not None
        
        # Step 5: Register indication router
        ric.register_indication_router(
            ran_function_id=2,
            xapp_name='anomaly_detector'
        )
        
        # Step 6: Simulate KPI indication
        indication = E2Indication(
            request_id=request_id,
            ran_function_id=2,
            action_id=1,
            indication_type='report',
            indication_header=b'header',
            indication_message=b'kpi_data'
        )
        
        ric.route_indication(indication)
        
        assert ric.statistics['indications_routed'] == 1
        
        # Step 7: Stop RIC
        ric.stop()
        assert ric.running == False
    
    def test_complete_dashboard_workflow(self, temp_db):
        """E2E: User login -> View targets -> Execute scan -> View results"""
        from falconone.utils.database import FalconOneDatabase
        
        db = FalconOneDatabase(db_path=temp_db, logger=Mock())
        
        # Step 1: Create user (admin)
        db.ensure_default_admin()
        
        # Step 2: Authenticate user
        user = db.verify_user('admin', 'admin')
        assert user is not None
        assert user['role'] == 'admin'
        
        # Step 3: Create targets (user action)
        target_ids = []
        for i in range(3):
            target_id = db.create_target(
                imsi=f'00101000000000{i:02d}',
                imei=f'12345678901234{i}'
            )
            target_ids.append(target_id)
        
        # Step 4: List all targets
        targets = db.list_targets()
        assert len(targets) >= 3
        
        # Step 5: View specific target
        target = db.get_target(target_ids[0])
        assert target is not None
        
        # Step 6: Simulate scan results
        for tid in target_ids:
            db.store_suci_capture(
                target_id=tid,
                suci=f'suci-0-001-01-0-0-{tid:08d}',
                capture_time=time.time(),
                network_id='00101',
                cell_id=12345
            )
        
        # Step 7: Export data (CSV format)
        export_data = []
        for tid in target_ids:
            target = db.get_target(tid)
            suci_captures = db.get_suci_captures_for_target(tid)
            export_data.append({
                'target': target,
                'captures': suci_captures
            })
        
        assert len(export_data) == 3


@pytest.mark.e2e
@pytest.mark.slow
class TestErrorRecoveryWorkflows:
    """E2E tests for error handling and recovery"""
    
    def test_database_recovery_workflow(self, temp_db):
        """E2E: Database corruption -> Backup -> Restore -> Verify"""
        from falconone.utils.database import FalconOneDatabase
        
        db = FalconOneDatabase(db_path=temp_db, logger=Mock())
        
        # Step 1: Create data
        target_id = db.create_target(
            imsi='001010000000999',
            imei='999999999999999'
        )
        
        # Step 2: Create backup
        backup_path = db.backup_database()
        assert os.path.exists(backup_path)
        
        # Step 3: Modify/delete data
        db.delete_target(target_id)
        
        # Verify deletion
        target = db.get_target(target_id)
        assert target is None or target.get('is_active') == 0
        
        # Step 4: Restore from backup
        db.restore_database(backup_path)
        
        # Step 5: Verify restoration
        target = db.get_target(target_id)
        assert target is not None
        assert target['imsi'] == '001010000000999'
        
        # Cleanup
        os.unlink(backup_path)
    
    def test_model_rollback_workflow(self):
        """E2E: Train model -> Detect drift -> Rollback -> Verify performance"""
        from falconone.ai.online_learning import OnlineLearner
        
        learner = OnlineLearner(logger=Mock())
        learner.initialize_model(n_features=5, n_classes=2)
        learner.warmup_samples = 10
        
        # Step 1: Train initial model
        X1 = np.random.rand(50, 5)
        y1 = (X1[:, 0] > 0.5).astype(int)
        learner.partial_fit(X1, y1, create_snapshot=True)
        
        initial_version = learner.current_version
        
        # Step 2: Train with more data
        X2 = np.random.rand(50, 5)
        y2 = (X2[:, 0] > 0.5).astype(int)
        learner.partial_fit(X2, y2, create_snapshot=True)
        
        # Step 3: Simulate drift (concept changes)
        X3 = np.random.rand(50, 5)
        y3 = (X3[:, 0] < 0.5).astype(int)  # Opposite relationship
        metrics = learner.partial_fit(X3, y3)
        
        # Step 4: Rollback if performance degrades
        if metrics['accuracy'] < 0.6:
            success = learner.rollback(version=initial_version)
            assert success == True
            assert learner.current_version == initial_version
        
        # Step 5: Verify model still works
        predictions = learner.predict(X1[:10])
        assert predictions is not None


@pytest.mark.e2e
class TestMultiUserWorkflows:
    """E2E tests for multi-user scenarios"""
    
    def test_concurrent_user_access(self, temp_db):
        """E2E: Multiple users accessing system concurrently"""
        from falconone.utils.database import FalconOneDatabase
        import threading
        
        db = FalconOneDatabase(db_path=temp_db, logger=Mock())
        
        # Create multiple users
        db.create_user('user1', 'pass1', 'operator', 'User One')
        db.create_user('user2', 'pass2', 'viewer', 'User Two')
        
        results = {'user1': None, 'user2': None}
        errors = []
        
        def user1_workflow():
            try:
                # User 1: Create targets
                for i in range(5):
                    db.create_target(
                        imsi=f'00101000000001{i:02d}',
                        imei=f'11111111111111{i}'
                    )
                results['user1'] = 'success'
            except Exception as e:
                errors.append(('user1', str(e)))
        
        def user2_workflow():
            try:
                # User 2: Read targets
                time.sleep(0.1)  # Let user1 create some
                targets = db.list_targets()
                results['user2'] = len(targets)
            except Exception as e:
                errors.append(('user2', str(e)))
        
        # Run concurrently
        t1 = threading.Thread(target=user1_workflow)
        t2 = threading.Thread(target=user2_workflow)
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert results['user1'] == 'success'
        assert results['user2'] >= 0


@pytest.mark.e2e
class TestPerformanceWorkflows:
    """E2E tests for performance characteristics"""
    
    def test_large_dataset_workflow(self, temp_db):
        """E2E: Handle large number of targets and captures"""
        from falconone.utils.database import FalconOneDatabase
        
        db = FalconOneDatabase(db_path=temp_db, logger=Mock())
        
        num_targets = 100
        captures_per_target = 10
        
        # Step 1: Create many targets
        target_ids = []
        start_time = time.time()
        
        for i in range(num_targets):
            target_id = db.create_target(
                imsi=f'001010{i:012d}',
                imei=f'{i:015d}'
            )
            target_ids.append(target_id)
        
        creation_time = time.time() - start_time
        assert creation_time < 10.0, f"Target creation too slow: {creation_time:.2f}s"
        
        # Step 2: Create many SUCI captures
        start_time = time.time()
        
        for target_id in target_ids[:10]:  # Test with subset
            for j in range(captures_per_target):
                db.store_suci_capture(
                    target_id=target_id,
                    suci=f'suci-0-001-01-0-0-{target_id:08d}-{j}',
                    capture_time=time.time(),
                    network_id='00101',
                    cell_id=12345 + j
                )
        
        capture_time = time.time() - start_time
        assert capture_time < 5.0, f"Capture storage too slow: {capture_time:.2f}s"
        
        # Step 3: Query performance
        start_time = time.time()
        targets = db.list_targets(limit=100)
        query_time = time.time() - start_time
        
        assert len(targets) >= num_targets
        assert query_time < 1.0, f"Query too slow: {query_time:.2f}s"
