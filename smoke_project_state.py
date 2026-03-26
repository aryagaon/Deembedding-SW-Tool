from pathlib import Path
import numpy as np

from rfdeembed import SParameterData, ProjectStateManager


def main():
    out_dir = Path("/mnt/user-data/outputs/rfdeembed_v1/state_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    freq = np.linspace(1e9, 2e9, 101)
    s = np.zeros((len(freq), 2, 2), dtype=complex)
    s[:, 0, 0] = 0.05
    s[:, 1, 1] = 0.04
    s[:, 1, 0] = 0.9 * np.exp(-1j * 2 * np.pi * freq * 0.5e-9)
    s[:, 0, 1] = 0.9 * np.exp(-1j * 2 * np.pi * freq * 0.5e-9)
    ntwk = SParameterData(freq_hz=freq, s=s, z0=50.0, name="test_network")

    mgr = ProjectStateManager()
    project_path = mgr.save_project(
        out_dir / "demo_project.json",
        networks={"test_network": ntwk},
        ui_state={"method": "Single-line TRL", "gate_mode": "bandstop"},
        latest_deembedded_name=None,
        project_name="demo_project",
    )

    networks, ui_state, latest_name, project_name = mgr.load_project(project_path)
    assert "test_network" in networks
    assert ui_state["method"] == "Single-line TRL"
    assert latest_name is None
    assert project_name == "demo_project"
    print("project_state_ok")
    print(project_path)


if __name__ == "__main__":
    main()
