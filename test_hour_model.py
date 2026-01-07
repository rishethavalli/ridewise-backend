import pickle
import sys

print("Testing hour_model.pkl...")
try:
    with open('hour_model.pkl', 'rb') as f:
        data = pickle.load(f)
    print(f"✓ SUCCESS - File loaded")
    print(f"  Type: {type(data)}")
except Exception as e:
    print(f"✗ ERROR: {str(e)[:300]}")
    import traceback
    traceback.print_exc()
