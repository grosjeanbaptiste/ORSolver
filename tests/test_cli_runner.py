import os
import sys
from io import StringIO
import pytest
from orsolver.cli.main import main


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "1.0.0" in captured.out or "1.0.0" in captured.err


def test_cli_solve_file(tmp_path, capsys):
    lp_content = """
    Maximize
      3 x1 + 5 x2
    Subject To
      x1 <= 4
      2 x2 <= 12
      3 x1 + 2 x2 <= 18
    End
    """
    model_file = tmp_path / "model.lp"
    model_file.write_text(lp_content)

    exit_code = main(["solve", str(model_file)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "OPTIMAL" in captured.out
    assert "36" in captured.out
    assert "x1" in captured.out
    assert "x2" in captured.out


def test_cli_solve_file_output(tmp_path):
    lp_content = """
    Minimize
      4 x1 + x2
    Subject To
      3 x1 + x2 == 3
      4 x1 + 3 x2 >= 6
      x1 + 2 x2 <= 4
    End
    """
    model_file = tmp_path / "model_min.lp"
    model_file.write_text(lp_content)
    output_file = tmp_path / "result.txt"

    exit_code = main(["solve", str(model_file), "--output", str(output_file)])
    assert exit_code == 0
    assert output_file.exists()
    content = output_file.read_text()
    assert "OPTIMAL" in content
    assert "3.4" in content


def test_cli_solve_json_output(tmp_path):
    import json
    lp_content = """
    Maximize
      3 x1 + 5 x2
    Subject To
      x1 <= 4
      2 x2 <= 12
    End
    """
    model_file = tmp_path / "model.lp"
    model_file.write_text(lp_content)
    json_out = tmp_path / "solution.json"

    exit_code = main(["solve", str(model_file), "-o", str(json_out)])
    assert exit_code == 0
    assert json_out.exists()

    data = json.loads(json_out.read_text())
    assert data["status"] == "OPTIMAL"
    assert data["objective_value"] == 42.0
    assert data["variables"]["x1"] == 4.0
    assert data["variables"]["x2"] == 6.0



def test_cli_solve_file_not_found(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["solve", "non_existent_file.lp"])
    assert exc_info.value.code != 0
