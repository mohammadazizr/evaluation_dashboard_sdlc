import os
import json
from qdrant_client import AsyncQdrantClient


async def export_collection_to_json(
    client: AsyncQdrantClient,
    collection_name: str,
    output_dir: str = "exports",
    batch_size: int = 200,
):
    """
    Fetch ALL points from a Qdrant collection and save payloads to a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_chunks = []
    offset = None

    while True:
        points, offset = await client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            all_chunks.append({
                "id": point.id,
                "payload": point.payload,
            })

        if offset is None:
            break

    output_path = os.path.join(output_dir, f"{collection_name}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    return output_path
