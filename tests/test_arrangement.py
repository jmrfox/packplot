from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from packplot import PackOptions, create_arrangement, load_arrangement, pack_images, save_arrangement
from tests.helpers import fast_opt_options


def _make_shape(path: Path, kind: str, color: tuple[int, int, int]) -> None:
    image = Image.new("RGBA", (40, 40), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    if kind == "rect":
        draw.rectangle((6, 12, 34, 28), fill=(*color, 255))
    elif kind == "tri":
        draw.polygon(((20, 5), (34, 34), (6, 34)), fill=(*color, 255))
    else:
        draw.ellipse((8, 8, 32, 32), fill=(*color, 255))
    image.save(path)


def test_save_and_replay_arrangement(tmp_path: Path) -> None:
    base_paths = [
        tmp_path / "cellA_mesh.png",
        tmp_path / "cellB_mesh.png",
        tmp_path / "cellC_mesh.png",
    ]
    _make_shape(base_paths[0], "rect", (230, 30, 30))
    _make_shape(base_paths[1], "tri", (30, 230, 30))
    _make_shape(base_paths[2], "ellipse", (30, 30, 230))

    base_result = pack_images(
        base_paths,
        options=fast_opt_options(target_aspect_ratio=1.4),
    )[0]
    arrangement = create_arrangement(base_result)
    arrangement_path = save_arrangement(arrangement, tmp_path / "layout.json")

    replay_paths = [
        tmp_path / "cellA_mesh.png",
        tmp_path / "cellB_mesh.png",
        tmp_path / "cellC_mesh.png",
    ]
    # Overwrite visuals but keep same IDs and silhouettes.
    _make_shape(replay_paths[0], "rect", (100, 100, 100))
    _make_shape(replay_paths[1], "tri", (140, 140, 140))
    _make_shape(replay_paths[2], "ellipse", (180, 180, 180))

    replay_result = pack_images(
        replay_paths,
        options=PackOptions(),
        arrangement=arrangement_path,
    )[0]

    assert replay_result.canvas_size == base_result.canvas_size
    assert len(replay_result.placements) == len(base_result.placements)
    for left, right in zip(base_result.placements, replay_result.placements):
        assert left.source_path.stem == right.source_path.stem
        assert left.top_left == right.top_left
        assert abs(left.angle_degrees - right.angle_degrees) < 1e-6

    loaded = load_arrangement(arrangement_path)
    assert loaded.canvas_size == arrangement.canvas_size
    assert len(loaded.entries) == len(arrangement.entries)


def test_arrangement_strict_mode_reports_missing_keys(tmp_path: Path) -> None:
    p1 = tmp_path / "one.png"
    p2 = tmp_path / "two.png"
    _make_shape(p1, "rect", (120, 10, 10))
    _make_shape(p2, "tri", (10, 120, 10))

    result = pack_images([p1, p2], options=fast_opt_options())[0]
    arrangement = create_arrangement(result)

    only_one = [p1]
    try:
        pack_images(only_one, options=PackOptions(), arrangement=arrangement, strict_arrangement=True)
        assert False, "Expected strict arrangement replay to fail for missing keys."
    except ValueError as exc:
        assert "missing" in str(exc).lower()


def test_arrangement_key_function_maps_modalities(tmp_path: Path) -> None:
    mesh_paths = [
        tmp_path / "cellA_mesh.png",
        tmp_path / "cellB_mesh.png",
        tmp_path / "cellC_mesh.png",
    ]
    _make_shape(mesh_paths[0], "rect", (210, 30, 30))
    _make_shape(mesh_paths[1], "tri", (30, 210, 30))
    _make_shape(mesh_paths[2], "ellipse", (30, 30, 210))

    key_fn = lambda p: p.stem.replace("_mesh", "").replace("_skeleton", "")
    base_result = pack_images(mesh_paths, options=fast_opt_options())[0]
    arrangement = create_arrangement(base_result, key_func=key_fn)

    skel_paths = [
        tmp_path / "cellA_skeleton.png",
        tmp_path / "cellB_skeleton.png",
        tmp_path / "cellC_skeleton.png",
    ]
    _make_shape(skel_paths[0], "rect", (120, 120, 120))
    _make_shape(skel_paths[1], "tri", (140, 140, 140))
    _make_shape(skel_paths[2], "ellipse", (180, 180, 180))

    replay = pack_images(
        skel_paths,
        options=PackOptions(),
        arrangement=arrangement,
        arrangement_key_func=key_fn,
        strict_arrangement=True,
    )[0]
    assert replay.canvas_size == base_result.canvas_size
    assert [p.top_left for p in replay.placements] == [p.top_left for p in base_result.placements]


def test_arrangement_duplicate_keys_raise(tmp_path: Path) -> None:
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    _make_shape(p1, "rect", (200, 40, 40))
    _make_shape(p2, "tri", (40, 200, 40))

    result = pack_images([p1, p2], options=fast_opt_options())[0]
    try:
        create_arrangement(result, key_func=lambda _p: "dup")
        assert False, "Expected duplicate arrangement keys to fail."
    except ValueError as exc:
        assert "duplicate arrangement keys" in str(exc).lower()


def test_apply_arrangement_duplicate_input_keys_raise(tmp_path: Path) -> None:
    mesh_paths = [
        tmp_path / "cellA_mesh.png",
        tmp_path / "cellB_mesh.png",
    ]
    _make_shape(mesh_paths[0], "rect", (210, 30, 30))
    _make_shape(mesh_paths[1], "tri", (30, 210, 30))
    arrangement = create_arrangement(pack_images(mesh_paths, options=fast_opt_options()))

    replay_paths = [
        tmp_path / "cellA_skeleton.png",
        tmp_path / "cellB_skeleton.png",
    ]
    _make_shape(replay_paths[0], "rect", (120, 120, 120))
    _make_shape(replay_paths[1], "tri", (140, 140, 140))

    try:
        pack_images(
            replay_paths,
            options=PackOptions(),
            arrangement=arrangement,
            arrangement_key_func=lambda _p: "dup",
            strict_arrangement=True,
        )
        assert False, "Expected duplicate keys during arrangement replay to fail."
    except ValueError as exc:
        assert "duplicate arrangement keys" in str(exc).lower()
