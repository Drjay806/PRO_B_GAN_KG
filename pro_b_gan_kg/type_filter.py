"""
Entity type filtering utilities for biological knowledge graphs.
Filters predictions by entity type based on ID prefixes.
"""

from typing import Dict, List, Set, Tuple


# ProHGT entity type mappings based on ID prefixes
ENTITY_TYPE_PREFIXES = {
    "protein": ["UniProt:"],
    "compound": ["CHEMBL:", "PubChem:"],
    "drug": ["DrugBank:", "DB"],
    "disease": ["DOID:", "MONDO:"],
    "go_term": ["GO:"],
    "pathway": ["Reactome:", "KEGG:", "R-"],
    "side_effect": ["SIDER:", "UMLS:"],
    "domain": ["InterPro:", "IPR"],
    "ec": ["EC:"],
}


def get_entity_type(entity_id: str) -> str:
    """
    Determine entity type from ID prefix.
    
    Args:
        entity_id: Entity identifier (e.g., "UniProt:P38398", "CHEMBL:25")
    
    Returns:
        Entity type name (e.g., "protein", "compound") or "unknown"
    """
    for entity_type, prefixes in ENTITY_TYPE_PREFIXES.items():
        for prefix in prefixes:
            if entity_id.startswith(prefix):
                return entity_type
    return "unknown"


def filter_candidates_by_type(
    candidates: List[Tuple[int, float]],
    id2entity: Dict[int, str],
    allowed_types: Set[str],
) -> List[Tuple[int, float]]:
    """
    Filter candidate predictions to only include specific entity types.
    
    Args:
        candidates: List of (entity_id, score) tuples
        id2entity: Mapping from internal ID to entity string
        allowed_types: Set of allowed types (e.g., {"protein", "compound"})
    
    Returns:
        Filtered list of candidates matching allowed types
    
    Example:
        >>> candidates = [(0, 0.95), (1, 0.87), (2, 0.82)]
        >>> id2entity = {0: "UniProt:P38398", 1: "CHEMBL:25", 2: "DOID:1234"}
        >>> filter_candidates_by_type(candidates, id2entity, {"protein"})
        [(0, 0.95)]
    """
    filtered = []
    for entity_idx, score in candidates:
        entity_id = id2entity.get(entity_idx, "")
        entity_type = get_entity_type(entity_id)
        if entity_type in allowed_types:
            filtered.append((entity_idx, score))
    return filtered


def filter_by_relation_signature(
    candidates: List[Tuple[int, float]],
    id2entity: Dict[int, str],
    relation: str,
    head_type: str,
) -> List[Tuple[int, float]]:
    """
    Filter candidates based on expected relation signature.
    
    Common biological relation signatures:
    - protein-protein interaction: protein -> protein
    - compound-target: compound -> protein
    - drug-disease: drug -> disease
    - protein-pathway: protein -> pathway
    - gene-phenotype: protein -> disease
    
    Args:
        candidates: List of (entity_id, score) tuples
        id2entity: Mapping from internal ID to entity string
        relation: Relation type name
        head_type: Type of the head entity
    
    Returns:
        Filtered candidates matching expected tail type for this relation
    """
    # Define relation signatures (head_type -> expected tail types)
    relation_signatures = {
        ("protein", "interacts_with"): {"protein"},
        ("protein", "participates_in"): {"pathway"},
        ("compound", "targets"): {"protein"},
        ("drug", "treats"): {"disease"},
        ("drug", "causes"): {"side_effect"},
        ("protein", "has_function"): {"go_term"},
        ("protein", "associated_with"): {"disease"},
        ("protein", "has_domain"): {"domain"},
        ("protein", "catalyzes"): {"ec"},
    }
    
    # Try to find matching signature
    expected_tail_types = relation_signatures.get((head_type, relation))
    
    if expected_tail_types is None:
        # No signature defined, return all candidates
        return candidates
    
    return filter_candidates_by_type(candidates, id2entity, expected_tail_types)


def get_type_statistics(
    candidates: List[Tuple[int, float]],
    id2entity: Dict[int, str],
) -> Dict[str, int]:
    """
    Get distribution of entity types in candidate predictions.
    
    Args:
        candidates: List of (entity_id, score) tuples
        id2entity: Mapping from internal ID to entity string
    
    Returns:
        Dictionary mapping entity type to count
    
    Example:
        >>> stats = get_type_statistics(candidates, id2entity)
        >>> print(stats)
        {'protein': 45, 'compound': 23, 'disease': 12}
    """
    type_counts = {}
    for entity_idx, _ in candidates:
        entity_id = id2entity.get(entity_idx, "")
        entity_type = get_entity_type(entity_id)
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
    return type_counts
