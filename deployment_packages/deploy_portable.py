#!/usr/bin/env python3
"""
Portable Archive Deployer
Extracts and sets up the MCP Agentic Workflow Accelerator from a portable archive.
"""

import os
import sys
import zipfile
import tarfile
import subprocess
import argparse
from pathlib import Path

def deploy_archive(archive_path: str, target_dir: str = None, auto_setup: bool = True):
    """Deploy the portable archive."""
    
    if not os.path.exists(archive_path):
        print(f"❌ Archive not found: {archive_path}")
        return False
    
    # Determine target directory
    if target_dir is None:
        archive_name = os.path.splitext(os.path.basename(archive_path))[0]
        target_dir = archive_name
    
    print(f"📦 Deploying MCP Agentic Workflow Accelerator")
    print(f"📁 Archive: {archive_path}")
    print(f"📁 Target: {target_dir}")
    print("=" * 50)
    
    try:
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Extract archive
        print("\n📦 Extracting archive...")
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(target_dir)
        elif archive_path.endswith('.tar.gz'):
            with tarfile.open(archive_path, 'r:gz') as tarf:
                tarf.extractall(target_dir)
        else:
            print(f"❌ Unsupported archive format: {archive_path}")
            return False
        
        print("✅ Archive extracted successfully")
        
        # Run setup if requested
        if auto_setup:
            print("\n🔧 Running automatic setup...")
            setup_success = run_setup(target_dir)
            
            if setup_success:
                print("\n✅ Deployment completed successfully!")
                print(f"\n📋 Next steps:")
                print(f"  cd {target_dir}")
                print(f"  python mcp_cli.py --help")
                print(f"  python test_system.py")
            else:
                print("\n⚠️  Deployment completed with setup warnings")
                print(f"\n📋 Manual setup required:")
                print(f"  cd {target_dir}")
                print(f"  pip install -r requirements.txt")
                print(f"  python test_system.py")
        else:
            print("\n✅ Deployment completed!")
            print(f"\n📋 Manual setup required:")
            print(f"  cd {target_dir}")
            print(f"  ./setup.sh  # Linux/Mac")
            print(f"  setup.bat   # Windows")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return False

def run_setup(target_dir: str) -> bool:
    """Run the setup script."""
    
    # Try to run setup script
    setup_script = os.path.join(target_dir, 'setup.sh')
    setup_bat = os.path.join(target_dir, 'setup.bat')
    
    try:
        if os.path.exists(setup_script) and os.access(setup_script, os.X_OK):
            # Linux/Mac setup
            result = subprocess.run(['bash', setup_script], 
                                  cwd=target_dir, 
                                  capture_output=True, 
                                  text=True)
            if result.returncode == 0:
                print("✅ Setup completed successfully")
                return True
            else:
                print(f"⚠️  Setup completed with warnings: {result.stderr}")
                return False
                
        elif os.path.exists(setup_bat):
            # Windows setup
            result = subprocess.run([setup_bat], 
                                  cwd=target_dir, 
                                  capture_output=True, 
                                  text=True, 
                                  shell=True)
            if result.returncode == 0:
                print("✅ Setup completed successfully")
                return True
            else:
                print(f"⚠️  Setup completed with warnings: {result.stderr}")
                return False
        else:
            # Manual setup
            print("📦 Installing dependencies...")
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                                  cwd=target_dir, 
                                  capture_output=True, 
                                  text=True)
            
            if result.returncode == 0:
                print("✅ Dependencies installed")
                
                # Test the system
                print("🧪 Testing system...")
                test_result = subprocess.run([sys.executable, 'test_system.py'], 
                                           cwd=target_dir, 
                                           capture_output=True, 
                                           text=True)
                
                if test_result.returncode == 0:
                    print("✅ System test passed")
                    return True
                else:
                    print(f"⚠️  System test failed: {test_result.stderr}")
                    return False
            else:
                print(f"❌ Failed to install dependencies: {result.stderr}")
                return False
                
    except Exception as e:
        print(f"⚠️  Setup failed: {e}")
        return False

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Deploy MCP Agentic Workflow Accelerator from portable archive')
    parser.add_argument('archive', help='Path to the portable archive')
    parser.add_argument('--target', '-t', help='Target directory for deployment')
    parser.add_argument('--no-setup', action='store_true', help='Skip automatic setup')
    
    args = parser.parse_args()
    
    success = deploy_archive(args.archive, args.target, not args.no_setup)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 