import builtins
from unittest.mock import patch
from eval.ragas_eval import run_ragas_evaluation


def test_missing_evaluator_cannot_pass_quality_gate():
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name in ('datasets', 'ragas'):
            raise ImportError('missing evaluator')
        return original(name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=missing):
        result = run_ragas_evaluation()
    assert result['status'] == 'BLOCKED'
    assert result['all_passed'] is False
    assert result['faithfulness'] is None
    assert result['samples_evaluated'] == 0
