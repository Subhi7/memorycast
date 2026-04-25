import asyncio, os, cognee
from dotenv import load_dotenv
load_dotenv()

def setup():
    cognee.config.set_llm_api_key(os.environ["OPENAI_API_KEY"])
    cognee.config.set_llm_model("gpt-4o-mini")
    cognee.config.set_embedding_provider("openai")
    cognee.config.set_embedding_model("text-embedding-3-small")
    cognee.config.set_embedding_dimensions(1536)

async def test_remember_recall():
    setup()

    # Prune old data
    await cognee.prune.prune_system(metadata=True)
    print("Pruned.")

    # Use new remember API
    lessons = [
        "AutoARIMA wins on high-volatility weak-seasonality finance series. volatility=0.18 seasonality=0.27 WAPE=9.4%",
        "AutoETS wins on low-volatility strong-seasonality retail series. volatility=0.06 seasonality=0.97 WAPE=1.6%",
        "GradientBoosting wins on high-volatility high-trend tech series. volatility=0.22 trend=0.91 WAPE=10.8%",
    ]
    for l in lessons:
        result = await cognee.remember(l)
        print(f"Remembered: {result}")

    print("\nSearching with recall...")
    for q in ["volatile finance series model", "seasonal retail model recommendation"]:
        r = await cognee.recall(q)
        print(f"  recall '{q}': type={type(r)} len={len(r) if r else 0}")
        if r:
            print(f"    first: {str(r[0])[:200]}")

    print("\nSearching with search CHUNKS...")
    for q in ["volatile finance", "seasonal retail"]:
        r = await cognee.search(q, query_type=cognee.SearchType.CHUNKS)
        print(f"  CHUNKS '{q}': {len(r) if r else 0} results")
        if r:
            print(f"    first: {str(r[0])[:200]}")

asyncio.run(test_remember_recall())
