from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict
import json

# 1. Load your dynamic data
with open('/home/cdsw/retrieve_developments/retrieve_070126/router_extraction/output/unique_level_names.json', 'r') as f:
    unique_level_names = json.load(f)

project_names = unique_level_names["project_names"]+["null"]

# 2. Dynamically create an Enum from the list
# We map the list items to (value, value) pairs
ProjectNameEnum = Enum("ProjectNameEnum", {name: name for name in project_names}, type=str)

class ProjectEntity(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    # 3. Use the Enum as the type
    project_name: List[ProjectNameEnum]
    project_id: List[str]
    childern_activity_uuids: List[str]

class EntityResponse(BaseModel):
    entity: ProjectEntity