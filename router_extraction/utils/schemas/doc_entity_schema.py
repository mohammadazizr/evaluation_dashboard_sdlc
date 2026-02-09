from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict
import json
import sys
from pathlib import Path

# Import config from project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
from config import UNIQUE_LEVEL_NAMES_PATH

# 1. Load your dynamic data
with open(UNIQUE_LEVEL_NAMES_PATH, 'r', encoding='utf-8') as f:
    unique_level_names = json.load(f)

doc_type_names = unique_level_names["doc_type_names"]+["null"]

DocTypeNameEnum = Enum("DocTypeNameEnum", {name: name for name in doc_type_names}, type=str)

class DocEntity(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    doc_type_name: List[DocTypeNameEnum]
    doc_id: List[str]
    childern_chunk_uuids: List[str]


class DocEntityResponse(BaseModel):
    entity: DocEntity
