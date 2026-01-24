import requests
import json
import os
from pathlib import Path

BASE_URL = "http://localhost:8000/api/v1"
SYMBOL = "AAPL"
MARKET = "us"

def test_cache_logic():
    print("=== Testing AI Analysis Cache Logic ===")
    
    # 1. 强制刷新 (Force Refresh) - 触发 LLM 调用并生成缓存
    print("\n[1] Requesting Force Refresh (Calling LLM)...")
    try:
        url = f"{BASE_URL}/analysis/auto/{SYMBOL}"
        params = {
            "market": MARKET,
            "use_llm": "true",
            "force_refresh": "true",
            "self_only": "true"
        }
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            print("Response received successfully")
            if data.get("llm_analysis"):
                print("✅ LLM Analysis received")
                print(f"Preview: {data['llm_analysis'][:100]}...")
            else:
                print("⚠️ LLM Analysis is empty (API might have failed internally but returned 200)")
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return
            
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    # 2. 检查缓存文件
    print("\n[2] Checking Cache Directory...")
    # Update path based on your local environment
    cache_dir = Path(r"c:\Users\ian28\Desktop\k-line-photo\data\llm_cache")
    
    found_cache = False
    if cache_dir.exists():
        for file in cache_dir.glob("*.json"):
            if SYMBOL in file.name and MARKET in file.name:
                print(f"✅ Cache file found: {file.name}")
                found_cache = True
                
                # 读取内容验证
                with open(file, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    print(f"   Created at: {content.get('created_at')}")
                break
    
    if not found_cache:
        print("❌ No cache file found for AAPL")
        # Proceed anyway to see API behavior
    
    # 3. 读取缓存 (Use Cache)
    print("\n[3] Requesting Cached Result...")
    try:
        params["force_refresh"] = "false"
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("llm_analysis"):
                print("✅ Cached Analysis received")
                # 检查后端日志或速度来确认是否命中缓存（这里主要看功能是否正常）
            else:
                print("⚠️ Analysis empty")
        else:
            print(f"❌ API Error: {response.status_code}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_cache_logic()
