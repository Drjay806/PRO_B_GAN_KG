import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class GoTermLoader:
    def __init__(self, go_data_path: Optional[Path] = None) -> None:
        self.go_data_path = go_data_path

    @staticmethod
    def fallback_go_terms() -> Dict[str, str]:
        logger.info("Using fallback GO term descriptions")
        return {
            "GO:0003674": "molecular_function basic molecular activities",
            "GO:0008150": "biological_process biological objective accomplished through gene expression",
            "GO:0005575": "cellular_component subcellular structures and protein complexes",
            "GO:0001882": "nucleoside binding binds to nucleosides or nucleotides",
            "GO:0005524": "ATP binding binds to adenosine triphosphate",
            "GO:0016301": "kinase activity catalyzes transfer of phosphate groups",
            "GO:0008234": "cysteine-type peptidase activity peptidase with cysteine in catalytic site",
            "GO:0006351": "transcription DNA-templated RNA synthesis directed by DNA",
            "GO:0006355": "regulation of transcription control of DNA-dependent RNA synthesis",
            "GO:0008289": "lipid binding binds to lipid substrates",
        }

    def load_go_terms(self, entity_ids: Dict[str, int]) -> Dict[int, str]:
        if self.go_data_path and self.go_data_path.exists():
            with self.go_data_path.open("r", encoding="utf-8") as f:
                term_id_to_text = json.load(f)
            logger.info(f"Loaded GO terms from {self.go_data_path}")
        else:
            term_id_to_text = self.fallback_go_terms()

        id_to_text = {}
        for entity_name, entity_id in entity_ids.items():
            if entity_name in term_id_to_text:
                id_to_text[entity_id] = term_id_to_text[entity_name]
            elif entity_name.startswith("GO:") and entity_name in term_id_to_text:
                id_to_text[entity_id] = term_id_to_text[entity_name]

        logger.info(f"Loaded text for {len(id_to_text)} / {len(entity_ids)} GO terms")
        return id_to_text


class PathwayLoader:
    def __init__(self, pathway_data_path: Optional[Path] = None) -> None:
        self.pathway_data_path = pathway_data_path

    @staticmethod
    def fallback_pathways() -> Dict[str, str]:
        logger.info("Using fallback pathway descriptions")
        return {
            "hsa00010": "Glycolysis major metabolic pathway glucose oxidation produces energy",
            "hsa00020": "Citric acid cycle central metabolic pathway energy production carbon oxidation",
            "hsa04010": "MAPK signaling pathway mitogen-activated protein kinase signaling cascade",
            "hsa04012": "ErbB signaling pathway receptor tyrosine kinase signaling growth differentiation",
            "hsa04014": "Ras signaling pathway small GTPase signaling cell proliferation differentiation",
        }

    def load_pathways(self, entity_ids: Dict[str, int]) -> Dict[int, str]:
        if self.pathway_data_path and self.pathway_data_path.exists():
            with self.pathway_data_path.open("r", encoding="utf-8") as f:
                pathway_id_to_text = json.load(f)
            logger.info(f"Loaded pathways from {self.pathway_data_path}")
        else:
            pathway_id_to_text = self.fallback_pathways()

        id_to_text = {}
        for entity_name, entity_id in entity_ids.items():
            if entity_name in pathway_id_to_text:
                id_to_text[entity_id] = pathway_id_to_text[entity_name]

        logger.info(f"Loaded text for {len(id_to_text)} / {len(entity_ids)} pathways")
        return id_to_text


class DiseaseLoader:
    def __init__(self, disease_data_path: Optional[Path] = None) -> None:
        self.disease_data_path = disease_data_path

    @staticmethod
    def fallback_diseases() -> Dict[str, str]:
        logger.info("Using fallback disease descriptions")
        return {
            "DOID:1816": "cancer malignant neoplasm uncontrolled cell proliferation growth",
            "DOID:2841": "asthma chronic airway disease inflammation reversible obstruction",
            "DOID:1826": "diabetes mellitus metabolic disorder hyperglycemia glucose intolerance",
            "DOID:4325": "heart disease cardiovascular pathology myocardial dysfunction",
            "DOID:14871": "Alzheimer disease neurodegenerative dementia cognitive decline amyloid",
        }

    def load_diseases(self, entity_ids: Dict[str, int]) -> Dict[int, str]:
        if self.disease_data_path and self.disease_data_path.exists():
            with self.disease_data_path.open("r", encoding="utf-8") as f:
                disease_id_to_text = json.load(f)
            logger.info(f"Loaded diseases from {self.disease_data_path}")
        else:
            disease_id_to_text = self.fallback_diseases()

        id_to_text = {}
        for entity_name, entity_id in entity_ids.items():
            if entity_name in disease_id_to_text:
                id_to_text[entity_id] = disease_id_to_text[entity_name]

        logger.info(f"Loaded text for {len(id_to_text)} / {len(entity_ids)} diseases")
        return id_to_text
