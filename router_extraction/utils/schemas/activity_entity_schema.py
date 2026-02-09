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

activity_names = unique_level_names["activity_names"]+["null"]

ActivityNameEnum = Enum("ActivityNameEnum", {name: name for name in activity_names}, type=str)

class ActivityEntity(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    activity_name: List[ActivityNameEnum]
    activity_id: List[str]
    childern_doc_type_uuids: List[str]


class ActivityEntityResponse(BaseModel):
    entity: ActivityEntity
