import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class BioKGTextExtractor:
    def __init__(self, metadata_dir: Optional[Path] = None) -> None:
        self.metadata_dir = metadata_dir
        self.entity_text: Dict[str, str] = {}
        self.fallback_text = self._load_fallback_text()

    @staticmethod
    def _load_fallback_text() -> Dict[str, str]:
        return {
            "protein_sample": "protein molecular entity that performs biological functions",
            "go_biological_process": "biological process system level function gene ontology term",
            "go_molecular_function": "molecular function biochemical activity gene ontology",
            "go_cellular_component": "cellular component subcellular structure protein complex",
            "pathway_apoptosis": "apoptosis programmed cell death pathway",
            "pathway_cancer": "cancer related pathway malignancy oncogenic transformation",
            "disease_human": "human disease genetic disorder illness",
            "disease_cancer": "cancer malignant neoplasm proliferation disorder",
            "side_effect_adverse": "side effect adverse drug reaction unwanted effect",
        }

    def load_entity_metadata(self, metadata_file: Path) -> Dict[str, str]:
        if metadata_file.exists():
            try:
                with metadata_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded metadata for {len(data)} entities")
                return data
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}, using fallback")
        return {}

    def get_entity_text(self, entity_id: str) -> str:
        if entity_id in self.entity_text:
            return self.entity_text[entity_id]

        if entity_id.startswith("protein"):
            return self.fallback_text.get("protein_sample", entity_id)
        elif entity_id.startswith("go_bp"):
            return self.fallback_text.get("go_biological_process", entity_id)
        elif entity_id.startswith("go_mf"):
            return self.fallback_text.get("go_molecular_function", entity_id)
        elif entity_id.startswith("go_cc"):
            return self.fallback_text.get("go_cellular_component", entity_id)
        elif entity_id.startswith("pathway"):
            return self.fallback_text.get("pathway_cancer", entity_id)
        elif entity_id.startswith("disease"):
            return self.fallback_text.get("disease_cancer", entity_id)
        elif entity_id.startswith("side_effect"):
            return self.fallback_text.get("side_effect_adverse", entity_id)
        else:
            return entity_id

    def extract_type_texts(self, entities: Dict[str, int], entity_type: str) -> Dict[int, str]:
        type_texts = {}
        for entity_name, entity_id in entities.items():
            if entity_name.startswith(entity_type):
                text = self.get_entity_text(entity_name)
                type_texts[entity_id] = text
        logger.info(f"Extracted text for {len(type_texts)} entities of type {entity_type}")
        return type_texts
