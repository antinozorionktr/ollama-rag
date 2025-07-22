#!/usr/bin/env python3
"""Quick test to debug config loading"""

import os
from dotenv import load_dotenv

print("🔍 Config Debug Test")
print("=" * 30)

# Test 1: Check if .env file exists
print("1. Checking .env file...")
if os.path.exists('.env'):
    print("✅ .env file exists")
    with open('.env', 'r') as f:
        content = f.read()
        print("📄 .env content:")
        print(content)
else:
    print("❌ .env file not found")

print("\n" + "-" * 30)

# Test 2: Check environment loading
print("2. Testing environment loading...")
load_dotenv()
model_from_env = os.getenv("OLLAMA_MODEL", "NOT_FOUND")
print(f"OLLAMA_MODEL from env: {model_from_env}")

print("\n" + "-" * 30)

# Test 3: Test config import
print("3. Testing config import...")
try:
    from config import Config
    print(f"Config.OLLAMA_MODEL: {Config.OLLAMA_MODEL}")
    print(f"Config.OLLAMA_BASE_URL: {Config.OLLAMA_BASE_URL}")
except Exception as e:
    print(f"❌ Config import error: {e}")

print("\n" + "-" * 30)

# Test 4: Force reload config
print("4. Force reloading config...")
try:
    import importlib
    import config
    importlib.reload(config)
    print(f"Reloaded Config.OLLAMA_MODEL: {config.Config.OLLAMA_MODEL}")
except Exception as e:
    print(f"❌ Reload error: {e}")

print("\n" + "-" * 30)

# Test 5: Test the smaller model directly
print("5. Testing gemma3:1b model...")
try:
    import ollama
    client = ollama.Client(host="http://localhost:11434")
    response = client.generate(model="gemma3:1b", prompt="Say hello briefly", stream=False)
    print(f"✅ gemma3:1b works: {response.get('response', '')[:100]}...")
except Exception as e:
    print(f"❌ gemma3:1b error: {e}")

print("\n" + "=" * 30)
print("✅ Debug complete!")