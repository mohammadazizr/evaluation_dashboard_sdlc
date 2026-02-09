from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict
import json


with open('/home/cdsw/retrieve_developments/retrieve_070126/router_extraction/output/unique_level_names.json', 'r') as f:
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
