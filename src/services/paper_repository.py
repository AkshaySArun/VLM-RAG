import json
import os
from threading import Lock
from typing import List, Dict, Any, Optional
from pathlib import Path

class PaperRepository:
    def __init__(self, storage_path: str = "./data/papers.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        if not self.storage_path.exists():
            self._save_all({})

    def _load_all(self) -> Dict[str, Any]:
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_all(self, data: Dict[str, Any]):
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_paper(self, paper_data: Dict[str, Any]):
        with self._lock:
            data = self._load_all()
            paper_id = paper_data["paper_id"]
            data[paper_id] = paper_data
            self._save_all(data)

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            data = self._load_all()
            return data.get(paper_id)

    def list_papers(self) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._load_all()
            return list(data.values())

    def update_paper(self, paper_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            data = self._load_all()
            if paper_id not in data:
                return None
            data[paper_id].update(updates)
            self._save_all(data)
            return data[paper_id]

    def delete_paper(self, paper_id: str) -> bool:
        with self._lock:
            data = self._load_all()
            if paper_id in data:
                del data[paper_id]
                self._save_all(data)
                return True
            return False
