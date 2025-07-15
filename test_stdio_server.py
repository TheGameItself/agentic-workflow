#!/usr/bin/env python3
"""
Test script for the MCP stdio server
"""
import subprocess
import json
import sys
import os

def test_stdio_server():
    """Test the stdio server with a simple request."""
    
    # Test request
    test_request = {
        "jsonrpc": "2.0",
        "method": "list_endpoints",
        "params": {},
        "id": 1
    }
    
    print("Testing MCP stdio server...")
    print(f"Request: {json.dumps(test_request)}")
    
    try:
        # Set up environment
        env = {
            **os.environ,
            "PYTHONPATH": f"{os.getcwd()}/src",
            "MCP_API_KEY": "test_key",
            "MCP_PROJECT_PATH": os.getcwd(),
            "MCP_VECTOR_BACKEND": "sqlitefaiss",
            "MCP_LOG_LEVEL": "WARNING"
        }
        
        # Run the stdio server
        result = subprocess.run(
            [sys.executable, "-m", "src.mcp.mcp_stdio"],
            input=json.dumps(test_request) + "\n",
            text=True,
            capture_output=True,
            timeout=10,
            env=env
        )
        
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout.strip()}")
        if result.stderr:
            print(f"Stderr: {result.stderr.strip()}")
        
        if result.returncode == 0:
            try:
                response = json.loads(result.stdout.strip())
                if "result" in response:
                    print("✅ Server responded successfully!")
                    print(f"Available endpoints: {len(response['result'])}")
                    return True
                else:
                    print("❌ Server returned error response")
                    print(f"Error: {response.get('error', 'Unknown error')}")
                    return False
            except json.JSONDecodeError:
                print("❌ Invalid JSON response")
                return False
        else:
            print("❌ Server failed to start or crashed")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Server timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_stdio_server()
    sys.exit(0 if success else 1) 