from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict
import json


with open('/home/cdsw/retrieve_developments/retrieve_070126/router_extraction/output/unique_level_names.json', 'r') as f:
    unique_level_names = json.load(f)

activity_names = unique_level_names["activity_names"]+["null"]

ActivityNameEnum = Enum("ActivityNameEnum", {name: name for name in activity_names}, type=str)

class ActivityEntity(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    activity_name: List[ActivityNameEnum]
    activity_id: List[str]
    childern_doc_type_uuids: List[str]


class ActivityEntityResponse(BaseModel):
    entity: ActivityEntity
