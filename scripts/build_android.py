#!/usr/bin/env python3
"""
Android APK Build Automation Script for MCP
- Builds debug and release APKs using Kivy, Python-for-Android, and Buildozer
- Signs APK if keystore is provided
- Outputs APK to dist/ directory

Usage:
    python scripts/build_android.py [--release] [--keystore <path>] [--alias <alias>] [--storepass <password>]
"""
import os
import sys
import subprocess
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Build Android APK for MCP using Kivy/Buildozer')
    parser.add_argument('--release', action='store_true', help='Build release APK (default: debug)')
    parser.add_argument('--keystore', type=str, help='Path to keystore for signing (release only)')
    parser.add_argument('--alias', type=str, help='Key alias for signing (release only)')
    parser.add_argument('--storepass', type=str, help='Keystore password (release only)')
    args = parser.parse_args()

    build_type = 'release' if args.release else 'debug'
    print(f"\nBuilding Android APK ({build_type})...")

    # Ensure buildozer is installed
    try:
        subprocess.run(['buildozer', '--version'], check=True)
    except Exception:
        print("Buildozer is not installed. Please install with: pip install buildozer kivy python-for-android")
        sys.exit(1)

    # Run buildozer build
    cmd = ['buildozer', f'android {build_type}']
    subprocess.run(cmd, check=True)

    # Sign APK if release and keystore provided
    if build_type == 'release' and args.keystore and args.alias and args.storepass:
        apk_path = Path('bin') / 'MCP-0.1-release-unsigned.apk'
        signed_apk_path = Path('dist') / 'MCP-0.1-release-signed.apk'
        print(f"Signing APK: {apk_path} -> {signed_apk_path}")
        sign_cmd = [
            'jarsigner',
            '-verbose',
            '-sigalg', 'SHA1withRSA',
            '-digestalg', 'SHA1',
            '-keystore', args.keystore,
            '-storepass', args.storepass,
            str(apk_path),
            args.alias
        ]
        subprocess.run(sign_cmd, check=True)
        signed_apk_path.parent.mkdir(exist_ok=True)
        apk_path.rename(signed_apk_path)
        print(f"Signed APK output: {signed_apk_path}")
    else:
        print("APK build complete. See bin/ or dist/ for output.")

if __name__ == '__main__':
    main() 