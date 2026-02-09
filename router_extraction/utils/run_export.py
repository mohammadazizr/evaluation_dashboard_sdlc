import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant.unset_proxy import unset_proxy

from .qdrant_exporter import export_collection_to_json

load_dotenv()


def main():
    """Main function untuk export collection dari Qdrant"""
    print(f"Connecting to Qdrant at {os.getenv('QDRANT_URL')}...")

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,
        verify=False
    )

    try:
        print("[OK] Connected to Qdrant successfully!")

        # Export one collection
        export_collection_to_json(client, "activity_coll")

        # Export multiple collections
        # for name in ["doc_type_coll", "proj_coll", "activity_coll"]:
        #     export_collection_to_json(client, name)

        print("[OK] Export completed successfully!")

    except Exception as e:
        print(f"[ERROR] Error during export: {e}")


if __name__ == "__main__":
    unset_proxy()
    main()
