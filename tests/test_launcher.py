from pathlib import Path

from jarvis.launcher import find_shortcut


def test_finds_the_least_nested_shortcut_case_insensitively(tmp_path: Path) -> None:
    user, common = tmp_path / "user", tmp_path / "common"
    (user / "Visual Studio Code").mkdir(parents=True)
    (user / "Visual Studio Code" / "Visual Studio Code.lnk").touch()
    (common / "Tools").mkdir(parents=True)
    (common / "visual studio code.LNK").touch()
    (common / "Tools" / "Other.lnk").touch()

    assert find_shortcut("Visual Studio Code.lnk", [user, common]) == common / "visual studio code.LNK"


def test_missing_shortcut_or_folder(tmp_path: Path) -> None:
    (tmp_path / "Other.lnk").touch()
    assert find_shortcut("Visual Studio Code.lnk", [tmp_path, tmp_path / "missing"]) is None
