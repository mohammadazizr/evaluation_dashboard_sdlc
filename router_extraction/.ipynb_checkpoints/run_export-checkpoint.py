import os
import asyncio
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant.unset_proxy import unset_proxy

from qdrant_exporter import export_collection_to_json

load_dotenv()


async def main():
    client = AsyncQdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,
    )

    try:
        # Export one collection
        await export_collection_to_json(client, "activity_coll")

        # Export multiple collections
        # for name in ["doc_type_coll", "proj_coll", "activity_coll"]:
        #     await export_collection_to_json(client, name)

    finally:
        await client.close()


if __name__ == "__main__":
    unset_proxy()
    asyncio.run(main())
