import sys
import subprocess
import os

def install_coverage():
    try:
        import coverage
        print("Coverage is already installed.")
    except ImportError:
        print("Installing coverage...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "coverage"])
            print("Coverage installed successfully.")
        except Exception as e:
            print(f"Error installing coverage: {e}")
            try:
                # Try alternative method
                import ensurepip
                ensurepip.bootstrap()
                subprocess.check_call([sys.executable, "-m", "pip", "install", "coverage"])
                print("Coverage installed successfully using ensurepip.")
            except Exception as e2:
                print(f"Error with alternative method: {e2}")

if __name__ == "__main__":
    install_coverage()