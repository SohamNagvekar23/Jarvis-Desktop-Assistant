from pathlib import Path

from jarvis.screenshot import save_new_file


def test_creates_missing_directory(tmp_path: Path) -> None:
    directory = tmp_path / "Pictures" / "Jarvis"
    path = save_new_file(b"png", directory, "shot", ".png")
    assert path == directory / "shot.png"
    assert path.read_bytes() == b"png"


def test_never_overwrites_existing_files(tmp_path: Path) -> None:
    (tmp_path / "shot.png").write_bytes(b"first")
    (tmp_path / "shot_1.png").write_bytes(b"second")

    path = save_new_file(b"third", tmp_path, "shot", ".png")

    assert path == tmp_path / "shot_2.png"
    assert path.read_bytes() == b"third"
    assert (tmp_path / "shot.png").read_bytes() == b"first"
    assert (tmp_path / "shot_1.png").read_bytes() == b"second"
