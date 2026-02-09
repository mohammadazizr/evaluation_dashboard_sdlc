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