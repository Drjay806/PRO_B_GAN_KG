import json
import logging
from pathlib import Path
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)


class GOTermFetcher:
    def __init__(self, cache_file: Optional[Path] = None) -> None:
        self.cache_file = cache_file
        self.cache: Dict[str, str] = {}
        if cache_file and cache_file.exists():
            with cache_file.open("r", encoding="utf-8") as f:
                self.cache = json.load(f)

    def fetch_definition(self, go_id: str) -> Optional[str]:
        if go_id in self.cache:
            return self.cache[go_id]

        try:
            url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{go_id}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "results" in data and len(data["results"]) > 0:
                    result = data["results"][0]
                    definition = result.get("definition", {}).get("text", result.get("name", ""))
                    self.cache[go_id] = definition
                    return definition
        except Exception as e:
            logger.warning(f"Failed to fetch GO {go_id}: {e}")

        return None

    def fetch_batch(self, go_ids: list[str]) -> Dict[str, str]:
        results = {}
        for go_id in go_ids:
            defn = self.fetch_definition(go_id)
            if defn:
                results[go_id] = defn
        return results

    def save_cache(self) -> None:
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_file.open("w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
            logger.info(f"Saved GO cache: {len(self.cache)} terms")


class PathwayFetcher:
    def __init__(self, cache_file: Optional[Path] = None) -> None:
        self.cache_file = cache_file
        self.cache: Dict[str, str] = {}
        if cache_file and cache_file.exists():
            with cache_file.open("r", encoding="utf-8") as f:
                self.cache = json.load(f)

    def fetch_pathway_name(self, pathway_id: str) -> Optional[str]:
        if pathway_id in self.cache:
            return self.cache[pathway_id]

        try:
            url = f"https://rest.kegg.jp/get/{pathway_id}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                lines = response.text.split("\n")
                for line in lines:
                    if line.startswith("NAME"):
                        description = line.split(None, 1)[1] if len(line.split()) > 1 else pathway_id
                        self.cache[pathway_id] = description
                        return description
        except Exception as e:
            logger.warning(f"Failed to fetch pathway {pathway_id}: {e}")

        return None

    def fetch_batch(self, pathway_ids: list[str]) -> Dict[str, str]:
        results = {}
        for pathway_id in pathway_ids:
            name = self.fetch_pathway_name(pathway_id)
            if name:
                results[pathway_id] = name
        return results

    def save_cache(self) -> None:
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_file.open("w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
            logger.info(f"Saved pathway cache: {len(self.cache)} pathways")


class DiseaseFetcher:
    def __init__(self, cache_file: Optional[Path] = None) -> None:
        self.cache_file = cache_file
        self.cache: Dict[str, str] = {}
        if cache_file and cache_file.exists():
            with cache_file.open("r", encoding="utf-8") as f:
                self.cache = json.load(f)

    def fetch_disease_name(self, disease_id: str) -> Optional[str]:
        if disease_id in self.cache:
            return self.cache[disease_id]

        try:
            url = f"https://www.ebi.ac.uk/ols/api/ontologies/mondo/terms?iri=http://purl.obolibrary.org/obo/{disease_id.replace(':', '_')}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "docs" in data and len(data["docs"]) > 0:
                    term = data["docs"][0]
                    description = term.get("description", [term.get("label", disease_id)])[0]
                    self.cache[disease_id] = description
                    return description
        except Exception as e:
            logger.warning(f"Failed to fetch disease {disease_id}: {e}")

        return None

    def fetch_batch(self, disease_ids: list[str]) -> Dict[str, str]:
        results = {}
        for disease_id in disease_ids:
            name = self.fetch_disease_name(disease_id)
            if name:
                results[disease_id] = name
        return results

    def save_cache(self) -> None:
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_file.open("w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
            logger.info(f"Saved disease cache: {len(self.cache)} diseases")


class ProteinFetcher:
    def __init__(self, cache_file: Optional[Path] = None) -> None:
        self.cache_file = cache_file
        self.cache: Dict[str, str] = {}
        if cache_file and cache_file.exists():
            with cache_file.open("r", encoding="utf-8") as f:
                self.cache = json.load(f)

    def fetch_protein_info(self, protein_id: str) -> Optional[str]:
        if protein_id in self.cache:
            return self.cache[protein_id]

        try:
            url = f"https://www.uniprot.org/api/uniprotkb/{protein_id}.json"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                protein_name = data.get("proteins", [{}])[0].get("recommendedName", {}).get("fullName", {}).get("value", "")
                description = data.get("comments", [{}])[0].get("texts", [{}])[0].get("value", protein_id)
                info = f"{protein_name} {description}".strip()
                self.cache[protein_id] = info
                return info
        except Exception as e:
            logger.warning(f"Failed to fetch protein {protein_id}: {e}")

        return None

    def fetch_batch(self, protein_ids: list[str]) -> Dict[str, str]:
        results = {}
        for protein_id in protein_ids:
            info = self.fetch_protein_info(protein_id)
            if info:
                results[protein_id] = info
        return results

    def save_cache(self) -> None:
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_file.open("w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
            logger.info(f"Saved protein cache: {len(self.cache)} proteins")
