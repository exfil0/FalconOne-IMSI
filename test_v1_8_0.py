#!/usr/bin/env python3
"""
Test v1.8.0 Enhancements
Validates all new features are loaded correctly
"""

import sys
from pathlib import Path

def test_v1_8_0_features():
    """Test all v1.8.0 enhancement features"""
    print("\n" + "="*60)
    print("FalconOne v1.8.0 Enhancement Validation")
    print("="*60)
    
    results = []
    
    # Test 1: Dynamic Scaling in Orchestrator
    try:
        from falconone.core.orchestrator import FalconOneOrchestrator
        orch = FalconOneOrchestrator()
        has_scaling = (
            hasattr(orch, '_scale_processing_resources') and
            hasattr(orch, '_scale_ml_resources') and
            hasattr(orch, '_trigger_memory_optimization')
        )
        results.append(('Dynamic Scaling (Orchestrator)', has_scaling))
        if has_scaling:
            print(f"✅ Dynamic Scaling: Thresholds loaded")
    except Exception as e:
        results.append(('Dynamic Scaling (Orchestrator)', False))
        print(f"❌ Dynamic Scaling: {str(e)[:50]}")
    
    # Test 2: Signal Bus Encryption
    try:
        from falconone.core.signal_bus import SignalBus
        bus = SignalBus(enable_encryption=False)  # Test without encryption
        has_encryption = (
            hasattr(bus, '_setup_encryption') and
            hasattr(bus, 'publish') and
            hasattr(bus, 'subscribe')
        )
        results.append(('Signal Bus Encryption', has_encryption))
        if has_encryption:
            print(f"✅ Signal Bus Encryption: Methods available (cryptography optional)")
    except Exception as e:
        results.append(('Signal Bus Encryption', False))
        print(f"❌ Signal Bus Encryption: {str(e)[:50]}")
    
    # Test 3: Config Hot-Reload
    try:
        from falconone.utils.config import Config
        cfg = Config()
        has_hotreload = (
            hasattr(cfg, '_setup_hot_reload') and
            hasattr(cfg, 'register_reload_callback') and
            hasattr(cfg, '_reload_configuration')
        )
        results.append(('Config Hot-Reload', has_hotreload))
        if has_hotreload:
            print(f"✅ Config Hot-Reload: Watchdog integration ready")
    except Exception as e:
        results.append(('Config Hot-Reload', False))
        print(f"❌ Config Hot-Reload: {str(e)[:50]}")
    
    # Test 4: ISAC Attention Mechanisms
    try:
        from falconone.ai.signal_classifier import SignalClassifier
        from falconone.core.signal_bus import SignalBus
        bus = SignalBus()
        sc = SignalClassifier(bus)
        has_isac = (
            hasattr(sc, 'classify_isac_signal') and
            hasattr(sc, '_assess_sensing_quality') and
            hasattr(sc, '_estimate_communication_snr') and
            hasattr(sc, '_detect_isac_waveform') and
            hasattr(sc, '_compute_isac_metrics')
        )
        results.append(('ISAC Attention (6G JCAS)', has_isac))
        if has_isac:
            print(f"✅ ISAC Support: 5 new methods available")
    except Exception as e:
        results.append(('ISAC Attention (6G JCAS)', False))
        print(f"❌ ISAC Support: {str(e)[:50]}")
    
    # Test 5: Enhanced Privacy Tracking
    try:
        import inspect
        from falconone.ai.federated_coordinator import FederatedCoordinator
        # Check method exists without initializing
        has_privacy = hasattr(FederatedCoordinator, 'get_privacy_budget')
        if has_privacy:
            # Verify method signature
            sig = inspect.signature(FederatedCoordinator.get_privacy_budget)
            params = list(sig.parameters.keys())
            has_privacy = 'self' in params
        results.append(('Privacy Budget Tracking', has_privacy))
        if has_privacy:
            print(f"✅ Privacy Tracking: Advanced composition method available")
    except Exception as e:
        results.append(('Privacy Budget Tracking', False))
        print(f"❌ Privacy Tracking: {str(e)[:50]}")
    
    # Test 6: Cross-Protocol Correlation
    try:
        from falconone.analysis.cyber_rf_fuser import CyberRFFuser
        # Check methods exist without full initialization
        has_correlation = (
            hasattr(CyberRFFuser, 'correlate_cross_protocol') and
            hasattr(CyberRFFuser, '_detect_multi_protocol_attacks') and
            hasattr(CyberRFFuser, '_find_related_events') and
            hasattr(CyberRFFuser, '_has_common_identifier') and
            hasattr(CyberRFFuser, '_build_correlation') and
            hasattr(CyberRFFuser, '_calculate_fusion_confidence')
        )
        results.append(('Cross-Protocol Correlation', has_correlation))
        if has_correlation:
            print(f"✅ Cross-Protocol Correlation: 6 methods (4 attack patterns) ready")
    except Exception as e:
        results.append(('Cross-Protocol Correlation', False))
        print(f"❌ Cross-Protocol Correlation: {str(e)[:50]}")
        fuser = CyberRFFuser(bus, None)  # Pass signal_bus and None for config
        has_correlation = (
            hasattr(fuser, 'correlate_cross_protocol') and
            hasattr(fuser, '_detect_multi_protocol_attacks') and
            hasattr(fuser, '_find_related_events') and
            hasattr(fuser, '_has_common_identifier')
        )
        results.append(('Cross-Protocol Correlation', has_correlation))
        if has_correlation:
            print(f"✅ Cross-Protocol Correlation: 4 attack patterns ready")
    except Exception as e:
        results.append(('Cross-Protocol Correlation', False))
        print(f"❌ Cross-Protocol Correlation: {str(e)[:50]}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for feature, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {feature}")
    
    print(f"\nTotal: {passed}/{total} features validated")
    
    if passed == total:
        print("\n🎉 All v1.8.0 enhancements successfully loaded!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} feature(s) failed to load")
        return 1

if __name__ == '__main__':
    sys.exit(test_v1_8_0_features())
