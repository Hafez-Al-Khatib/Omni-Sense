"""
Create minimal placeholder ONNX models for classifiers.
These are empty stubs to allow the app to start. Replace with real trained models later.
"""
import sys
from pathlib import Path

models_dir = Path(__file__).parent.parent / "iep2" / "models"
models_dir.mkdir(exist_ok=True)

def create_minimal_onnx(output_path: Path):
    """Create a minimal valid ONNX model."""
    try:
        from onnx import helper, TensorProto, save_model
        
        # Create minimal inputs/outputs
        X = helper.make_tensor_value_info('float_input', TensorProto.FLOAT, [None, 10])
        Y = helper.make_tensor_value_info('output', TensorProto.INT64, [None])
        
        # Create a simple Cast node
        node = helper.make_node(
            'Cast',
            inputs=['float_input'],
            outputs=['output'],
            to=TensorProto.INT64
        )
        
        # Create the graph
        graph = helper.make_graph(
            [node],
            'placeholder',
            [X],
            [Y]
        )
        
        # Create model
        model = helper.make_model(graph, producer_name='omni-sense')
        model.opset_import[0].version = 13
        
        save_model(model, str(output_path))
        return True
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return False

print("Creating placeholder ONNX models...")

if create_minimal_onnx(models_dir / "rf_classifier.onnx"):
    print(f"✓ Created: rf_classifier.onnx")
else:
    print("✗ Failed to create rf_classifier.onnx")

if create_minimal_onnx(models_dir / "xgboost_classifier.onnx"):
    print(f"✓ Created: xgboost_classifier.onnx")
else:
    print("✗ Failed to create xgboost_classifier.onnx")

print("Done!")

