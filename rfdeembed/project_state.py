from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional
import json

from .sparameter_data import SParameterData


@dataclass
class ProjectSnapshot:
    project_name: str
    assets_dir: str
    networks: list[dict[str, Any]]
    latest_deembedded_name: Optional[str]
    ui_state: dict[str, Any]


class ProjectStateManager:
    """
    Save/load project state to JSON plus Touchstone assets.

    Design choice for v1:
    - every in-memory network is exported as a .s1p/.s2p asset in a sibling assets folder
    - project JSON stores UI selections and project metadata
    - loading reconstructs all networks from the saved assets, including gated/de-embedded results
    """

    def save_project(
        self,
        project_file: str | Path,
        networks: Dict[str, SParameterData],
        ui_state: dict[str, Any],
        latest_deembedded_name: Optional[str] = None,
        project_name: Optional[str] = None,
    ) -> Path:
        project_path = Path(project_file)
        if project_path.suffix.lower() != ".json":
            project_path = project_path.with_suffix(".json")
        project_path.parent.mkdir(parents=True, exist_ok=True)

        assets_dir = project_path.with_suffix("")
        assets_dir = assets_dir.parent / f"{assets_dir.name}_assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        network_entries: list[dict[str, Any]] = []
        for name, ntwk in networks.items():
            safe_name = self._sanitize_filename(name)
            asset_file = assets_dir / f"{safe_name}.s{ntwk.n_ports}p"
            ntwk.to_touchstone(asset_file)
            network_entries.append(
                {
                    "name": name,
                    "asset_path": str(asset_file.name),
                    "n_ports": ntwk.n_ports,
                    "z0": float(ntwk.z0.real) if hasattr(ntwk.z0, "real") else float(ntwk.z0),
                    "metadata": self._jsonable(ntwk.metadata),
                }
            )

        snapshot = ProjectSnapshot(
            project_name=project_name or project_path.stem,
            assets_dir=str(assets_dir.name),
            networks=network_entries,
            latest_deembedded_name=latest_deembedded_name,
            ui_state=self._jsonable(ui_state),
        )

        with project_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(snapshot), f, indent=2)

        return project_path

    def load_project(self, project_file: str | Path) -> tuple[Dict[str, SParameterData], dict[str, Any], Optional[str], str]:
        project_path = Path(project_file)
        with project_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        assets_dir = project_path.parent / data["assets_dir"]
        networks: Dict[str, SParameterData] = {}
        for entry in data["networks"]:
            asset_path = assets_dir / entry["asset_path"]
            ntwk = SParameterData.from_touchstone(asset_path)
            ntwk.name = entry["name"]
            if isinstance(entry.get("metadata"), dict):
                ntwk.metadata.update(entry["metadata"])
            networks[ntwk.name] = ntwk

        ui_state = data.get("ui_state", {})
        latest_deembedded_name = data.get("latest_deembedded_name")
        project_name = data.get("project_name", project_path.stem)
        return networks, ui_state, latest_deembedded_name, project_name

    def _sanitize_filename(self, name: str) -> str:
        keep = []
        for ch in name:
            if ch.isalnum() or ch in ("-", "_", "."):
                keep.append(ch)
            else:
                keep.append("_")
        return "".join(keep).strip("_") or "network"

    def _jsonable(self, obj: Any) -> Any:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {str(k): self._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._jsonable(v) for v in obj]
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
        except Exception:
            pass
        return str(obj)
