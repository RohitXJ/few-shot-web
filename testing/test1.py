import os
import sys
import importlib.util
import torch

# Step 1: Setup project root so we can import main.py and utils.*
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

# Step 2: Load main.py dynamically
main_path = os.path.join(ROOT_DIR, "main.py")
spec = importlib.util.spec_from_file_location("main", main_path)
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)

# Step 3: Define a default config for testing
test_config = {
    "support_dir": "data/support/",
    "query_dir": "data/query/",
    "backbone": "resnet34"
}

# Step 4: Run the few-shot pipeline
result = main.run_fewshot_pipeline(test_config)

# Step 5: Assertions
assert isinstance(result, dict), "Result should be a dictionary."
assert "export_path" in result, "Missing 'export_path' in result."
assert "accuracy" in result, "Missing 'accuracy' in result."
assert os.path.exists(result["export_path"]), f"Exported model file not found: {result['export_path']}"

print("✅ All basic tests passed successfully!")
