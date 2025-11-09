"""
Diagnostic script to identify the import issue
"""

import sys
import os

print("="*60)
print("DIAGNOSTIC SCRIPT")
print("="*60)

# Check 1: Current directory
print("\n1. Current Directory:")
print(f"   {os.getcwd()}")

# Check 2: Files in directory
print("\n2. Files in current directory:")
files = os.listdir('.')
for f in files:
    if f.endswith('.py'):
        print(f"   ✓ {f}")

# Check 3: Check if environment.py exists
if 'environment.py' in files:
    print("\n3. ✅ environment.py found")
    
    # Check file size
    size = os.path.getsize('environment.py')
    print(f"   File size: {size} bytes")
    
    if size < 100:
        print("   ⚠️ File seems too small! Make sure you copied the full code.")
else:
    print("\n3. ❌ environment.py NOT FOUND!")
    print("   Create environment.py with the code from the artifact.")
    sys.exit(1)

# Check 4: Try importing
print("\n4. Testing import...")
try:
    import environment
    print(f"   ✅ Import successful")
    print(f"   Type: {type(environment)}")
    print(f"   Has DetectiveEnv: {'DetectiveEnv' in dir(environment)}")
    
    # Check 5: Try to instantiate
    print("\n5. Testing DetectiveEnv class...")
    if hasattr(environment, 'DetectiveEnv'):
        DetectiveEnv = environment.DetectiveEnv
        print(f"   DetectiveEnv type: {type(DetectiveEnv)}")
        
        # Try to create instance
        try:
            env = DetectiveEnv()
            print(f"   ✅ Environment created successfully!")
            print(f"   Environment type: {type(env)}")
        except Exception as e:
            print(f"   ❌ Failed to create environment: {e}")
    else:
        print(f"   ❌ DetectiveEnv class not found in environment module")
        print(f"   Available: {dir(environment)}")
        
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Check 6: Gymnasium installation
print("\n6. Checking dependencies...")
try:
    import gymnasium
    print(f"   ✅ gymnasium installed (version {gymnasium.__version__})")
except ImportError:
    print(f"   ❌ gymnasium not installed - run: pip install gymnasium")

try:
    import numpy
    print(f"   ✅ numpy installed (version {numpy.__version__})")
except ImportError:
    print(f"   ❌ numpy not installed - run: pip install numpy")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)