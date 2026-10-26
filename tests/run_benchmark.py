import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.retrieval.search import SemanticSearcher

def run_tests():
    searcher = SemanticSearcher()

    queries = [
        "Ancient Egyptian bronze sculpture or gold jewelry",
        "French impressionist oil painting with water or landscape",
        "Japanese samurai armor or ceremonial katana",
        "Renaissance Madonna and child altarpiece",
    ]

    print("\n" + "="*70)
    print("MUSEGUIDE SEMANTIC SEARCH BENCHMARK")
    print("="*70)

    for q in queries:
        print(f"\n>> Query: \"{q}\"")
        results = searcher.search(q, top_k=2)
        for r in results:
            print(f"   #{r['rank']} [Score: {r['score']:.4f}] {r['title']}")
            print(f"      Culture: {r['culture']} | Dept: {r['department']}")
            print(f"      Creator: {r['creator_name']} | Date: {r['creation_date']}")
            print(f"      URL: {r['museum_url']}")

    # Test Department Filter
    print("\n" + "="*70)
    print("TEST: DEPARTMENT FILTER (Japanese Art only)")
    print("="*70)
    filtered = searcher.search("warrior battle", top_k=2, department="Japanese Art")
    for r in filtered:
        print(f"   #{r['rank']} [Score: {r['score']:.4f}] {r['title']}")
        print(f"      Dept: {r['department']} | Culture: {r['culture']}")

    # Test Artefact-to-Artefact Similarity Discovery
    print("\n" + "="*70)
    print("TEST: ARTEFACT SIMILARITY DISCOVERY (Find related items)")
    print("="*70)
    # Search for an anchor artwork
    seed = searcher.search("Greek amphora vase pottery", top_k=1)[0]
    print(f"Anchor Artefact: \"{seed['title']}\" (ID: {seed['id']})")
    print(f"Anchor Culture: {seed['culture']} | Dept: {seed['department']}")

    similar = searcher.find_similar(seed['id'], top_k=3)
    print("\nTop 3 Semantically Related Artefacts:")
    for r in similar:
        print(f"   #{r['rank']} [Similarity: {r['score']:.4f}] {r['title']}")
        print(f"      Culture: {r['culture']} | Dept: {r['department']}")
        print(f"      Date: {r['creation_date']}")

    print("\n" + "="*70)
    print("All benchmark retrieval tests completed successfully!")
    print("="*70)

if __name__ == "__main__":
    run_tests()
