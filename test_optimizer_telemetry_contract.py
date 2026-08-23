import ast
from pathlib import Path


def test_optimizer_telemetry_preserves_minibatch_metric_shape():
    source = Path("vendor/verl/verl/workers/engine/base.py").read_text()
    tree = ast.parse(source)
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Subscript):
            continue
        rendered = ast.unparse(target)
        if rendered == "outputs['metrics'][name]":
            matches.append(node.value)
    assert len(matches) == 1
    value = matches[0]
    assert isinstance(value, ast.List)
    assert len(value.elts) == 1 and isinstance(value.elts[0], ast.Name)
    assert value.elts[0].id == "value"
