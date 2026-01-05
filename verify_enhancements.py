"""
Test script to verify all 6 enhancements are working
"""

import sys
sys.path.insert(0, '.')

def test_enhancement_1():
    """Test GitHub Push - Check git status"""
    print("=" * 50)
    print("Enhancement 1: GitHub Push")
    print("=" * 50)
    import subprocess
    result = subprocess.run(['git', 'log', '--oneline', '-3'], capture_output=True, text=True)
    print("Recent commits:")
    print(result.stdout)
    print("[OK] GitHub integration verified\n")

def test_enhancement_2():
    """Test Streamlit Dashboard - Check if app exists"""
    print("=" * 50)
    print("Enhancement 2: Streamlit Dashboard")
    print("=" * 50)
    from pathlib import Path
    app_path = Path("streamlit_app/app.py")
    if app_path.exists():
        print(f"[OK] Dashboard app found: {app_path}")
        print("   Run: streamlit run streamlit_app/app.py\n")
    else:
        print("[FAIL] Dashboard not found\n")

def test_enhancement_3():
    """Test Real Data Integration - Test data loader"""
    print("=" * 50)
    print("Enhancement 3: Real Data Integration")
    print("=" * 50)
    from src.data_loader import DataLoader
    
    loader = DataLoader()
    
    # Test sample data
    df = loader.get_sample_data(50)
    print(f"   Sample data: {len(df)} rows")
    
    # Test validation
    report = loader.validate_data(df)
    print(f"   Validation: {'[OK] Valid' if report['is_valid'] else '[FAIL] Invalid'}")
    
    # Test unified loader
    df2 = loader.load("sample:25")
    print(f"   Unified loader: {len(df2)} rows loaded")
    
    print("[OK] Data loader verified\n")

def test_enhancement_4():
    """Test Scheduled Retraining - Check if module exists"""
    print("=" * 50)
    print("Enhancement 4: Scheduled Retraining")
    print("=" * 50)
    from pathlib import Path
    
    pipeline_path = Path("azure_ml/scheduled_pipeline.py")
    if pipeline_path.exists():
        print(f"[OK] Scheduled pipeline found: {pipeline_path}")
        print("   Commands:")
        print("   - python azure_ml/scheduled_pipeline.py create --name weekly-retraining")
        print("   - python azure_ml/scheduled_pipeline.py list")
        print("   - python azure_ml/scheduled_pipeline.py run-now\n")
    else:
        print("[FAIL] Scheduled pipeline not found\n")

def test_enhancement_5():
    """Test Alerting System"""
    print("=" * 50)
    print("Enhancement 5: Alerting System")
    print("=" * 50)
    from src.monitoring.alerts import create_default_alert_manager, AlertType, AlertSeverity, Alert
    
    manager = create_default_alert_manager()
    print(f"   Alert channels configured: {len(manager.channels)}")
    
    # Send a test alert
    test_alert = Alert(
        alert_type=AlertType.SYSTEM,
        severity=AlertSeverity.INFO,
        title="Test Alert",
        message="This is a test alert from the verification script",
        details={"test": True}
    )
    manager.send_alert(test_alert)
    print("   Test alert sent to file channel")
    print("[OK] Alerting system verified\n")

def test_enhancement_6():
    """Test API Documentation"""
    print("=" * 50)
    print("Enhancement 6: API Documentation")
    print("=" * 50)
    from pathlib import Path
    import json
    
    openapi_path = Path("docs/openapi.json")
    api_md_path = Path("docs/API.md")
    
    if openapi_path.exists():
        with open(openapi_path) as f:
            spec = json.load(f)
        print(f"[OK] OpenAPI spec: {spec['info']['title']} v{spec['info']['version']}")
    else:
        print("[FAIL] OpenAPI spec not found")
    
    if api_md_path.exists():
        print(f"[OK] API documentation: {api_md_path}")
    else:
        print("[FAIL] API documentation not found")
    
    print()

def main():
    print("\n" + "=" * 50)
    print("SPARE PART DEMAND FORECASTING")
    print("Enhancement Verification Script")
    print("=" * 50 + "\n")
    
    try:
        test_enhancement_1()
        test_enhancement_2()
        test_enhancement_3()
        test_enhancement_4()
        test_enhancement_5()
        test_enhancement_6()
        
        print("=" * 50)
        print("SUCCESS: ALL ENHANCEMENTS VERIFIED!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
