#!/usr/bin/env python3
"""
Production Readiness Check Script
Verifies all components are ready for production deployment
"""

import sys
import os
from pathlib import Path

def check_files():
    """Check if all required files exist."""
    required_files = [
        'streamlit_app/app.py',
        'requirements.txt',
        'README.md',
        'PRODUCTION_DEPLOYMENT.md',
        'docs/API.md',
        'docs/openapi.json',
        'src/monitoring/drift_monitor.py',
        'src/monitoring/alerts.py',
        'azure_ml/deploy_model.py',
        'azure_ml/scheduled_pipeline.py',
        '.gitignore'
    ]
    
    print("=" * 60)
    print("CHECKING REQUIRED FILES")
    print("=" * 60)
    
    missing = []
    for file in required_files:
        if Path(file).exists():
            print(f"[OK] {file}")
        else:
            print(f"[FAIL] {file}")
            missing.append(file)
    
    return len(missing) == 0

def check_environment():
    """Check environment variables."""
    print("\n" + "=" * 60)
    print("CHECKING ENVIRONMENT VARIABLES")
    print("=" * 60)
    
    required_vars = [
        'AZURE_ML_ENDPOINT_URL',
        'AZURE_ML_API_KEY',
    ]
    
    optional_vars = [
        'AZURE_STORAGE_CONNECTION_STRING',
        'SMTP_SERVER',
        'TEAMS_WEBHOOK_URL',
        'SLACK_WEBHOOK_URL'
    ]
    
    from dotenv import load_dotenv
    load_dotenv()
    
    missing_required = []
    for var in required_vars:
        if os.getenv(var):
            print(f"[OK] {var}")
        else:
            print(f"[FAIL] {var} (REQUIRED)")
            missing_required.append(var)
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"[OK] {var}")
        else:
            print(f"[WARN] {var} (optional)")
    
    return len(missing_required) == 0

def check_dependencies():
    """Check if all dependencies are installed."""
    print("\n" + "=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)
    
    try:
        import streamlit
        print(f"[OK] streamlit ({streamlit.__version__})")
    except ImportError:
        print("[FAIL] streamlit")
        return False
    
    try:
        import pandas
        print(f"[OK] pandas ({pandas.__version__})")
    except ImportError:
        print("[FAIL] pandas")
        return False
    
    try:
        import plotly
        print(f"[OK] plotly ({plotly.__version__})")
    except ImportError:
        print("[FAIL] plotly")
        return False
    
    try:
        import requests
        print(f"[OK] requests ({requests.__version__})")
    except ImportError:
        print("[FAIL] requests")
        return False
    
    return True

def check_git_status():
    """Check if repo is clean and pushed."""
    print("\n" + "=" * 60)
    print("CHECKING GIT STATUS")
    print("=" * 60)
    
    import subprocess
    
    # Check uncommitted changes
    result = subprocess.run(['git', 'status', '--porcelain'], 
                          capture_output=True, text=True)
    
    if result.stdout.strip():
        print("[WARN] Uncommitted changes found:")
        print(result.stdout)
        return False
    else:
        print("[OK] No uncommitted changes")
    
    # Check if pushed
    result = subprocess.run(['git', 'log', 'origin/main..HEAD', '--oneline'], 
                          capture_output=True, text=True)
    
    if result.stdout.strip():
        print("[WARN] Unpushed commits found:")
        print(result.stdout)
        return False
    else:
        print("[OK] All commits pushed to remote")
    
    return True

def check_ci_cd():
    """Check CI/CD status."""
    print("\n" + "=" * 60)
    print("CHECKING CI/CD STATUS")
    print("=" * 60)
    
    if Path('.github/workflows/ci.yml').exists():
        print("[OK] CI pipeline configured")
    else:
        print("[FAIL] CI pipeline not configured")
        return False
    
    if Path('.github/workflows/cd.yml').exists():
        print("[OK] CD pipeline configured")
    else:
        print("[FAIL] CD pipeline not configured")
        return False
    
    print("[INFO] Check GitHub Actions for latest status:")
    print("   https://github.com/shan31/spare-part-demand-forecasting/actions")
    
    return True

def main():
    """Run all checks."""
    print("\n")
    print("=" * 60)
    print(" " * 10 + "PRODUCTION READINESS CHECK")
    print(" " * 10 + "Spare Part Demand Forecasting")
    print("=" * 60)
    print("\n")
    
    checks = {
        "Required Files": check_files(),
        "Environment Variables": check_environment(),
        "Dependencies": check_dependencies(),
        "Git Status": check_git_status(),
        "CI/CD Configuration": check_ci_cd()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{check_name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n[SUCCESS] All checks passed! Ready for production deployment.")
        print("\nNext steps:")
        print("  1. Review PRODUCTION_DEPLOYMENT.md")
        print("  2. Choose deployment option (Streamlit Cloud, Azure, or Docker)")
        print("  3. Deploy and monitor!")
        return 0
    else:
        print("\n[WARNING] Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
