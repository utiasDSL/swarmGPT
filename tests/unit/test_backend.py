from conftest import virtual_crazyswarm_config

from swarm_gpt.core.backend import AppBackend


def test_backend_init():
    config_path = virtual_crazyswarm_config(n_drones=4)
    app = AppBackend(config_path)
    assert app.choreographer.num_drones == 4
    assert app.choreographer.messages == []


def test_songs():
    config_path = virtual_crazyswarm_config(n_drones=4)
    app = AppBackend(config_path)
    assert isinstance(app.songs, list)
    available_songs = [s.stem for s in app.music_manager.music_dir.glob("*.mp3")]
    for song in app.songs:
        assert isinstance(song, str), f"Song {song} is not a string"
        assert song in available_songs, f"Song {song} is not in the available songs"


def test_presets():
    config_path = virtual_crazyswarm_config(n_drones=4)
    app = AppBackend(config_path)
    assert isinstance(app.presets, list)
    for preset in app.presets:
        assert isinstance(preset, str), f"Preset {preset} is not a string"
