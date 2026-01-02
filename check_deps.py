import importlib

deps = [
    ('tensorflow', 'TensorFlow'),
    ('torch', 'PyTorch'), 
    ('transformers', 'Transformers'),
    ('ray', 'Ray'),
    ('gym', 'Gym'),
    ('pytest', 'Pytest')
]

print("Checking AI/ML Dependencies:")
print("-" * 50)

for module_name, display_name in deps:
    try:
        m = importlib.import_module(module_name)
        version = getattr(m, '__version__', 'unknown')
        print(f"✅ {display_name:15} : {version}")
    except ImportError:
        print(f"❌ {display_name:15} : Not installed")
