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
import shutil

    cmd = ['buildozer', f'android {build_type}']
    try:
    subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building APK: {e}")
        return False

        apk_path = Path('bin') / 'MCP-0.1-release-unsigned.apk'
        signed_apk_path = Path('dist') / 'MCP-0.1-release-signed.apk'
    
    if not apk_path.exists():
        print(f"Error: Unsigned APK not found at {apk_path}")
        return False
        
        print(f"Signing APK: {apk_path} -> {signed_apk_path}")
    
        sign_cmd = [
            'jarsigner',
            '-verbose',
            '-sigalg', 'SHA1withRSA',
            '-digestalg', 'SHA1',
            str(apk_path),
        alias
        ]
    
    try:
        subprocess.run(sign_cmd, check=True)
        signed_apk_path.parent.mkdir(exist_ok=True)
        shutil.copy2(apk_path, signed_apk_path)
        print(f"Signed APK output: {signed_apk_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error signing APK: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Build Android APK for MCP using Kivy/Buildozer')
    parser.add_argument('--release', action='store_true', help='Build release APK (default: debug)')
    parser.add_argument('--keystore', type=str, help='Path to keystore for signing (release only)')
    parser.add_argument('--alias', type=str, help='Key alias for signing (release only)')
    parser.add_argument('--storepass', type=str, help='Keystore password (release only)')
    args = parser.parse_args()

    build_type = 'release' if args.release else 'debug'
    
    if not check_buildozer():
        sys.exit(1)
        
    if not build_apk(build_type):
        sys.exit(1)

    # Sign APK if release and keystore provided
    if build_type == 'release' and args.keystore and args.alias and args.storepass:
        if not sign_apk(args.keystore, args.alias, args.storepass):
            sys.exit(1)
    else:
        print("APK build complete. See bin/ or dist/ for output.")

if __name__ == '__main__':
    main() 