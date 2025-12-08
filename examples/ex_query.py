from kg_ai_papers.api.query import explain_paper

result = explain_paper("1706.03762")  # Attention Is All You Need

print("Paper:", result.paper.title)
print("\nTop concepts:")
for c in result.concepts:
    print(f"  - {c.label} (weight={c.weight:.3f})")

print("\nMost influential references:")
for ref in result.influential_references:
    print(f"  -> {ref.title}  [influence={ref.influence_score:.3f}, sim={ref.similarity:.3f}]")

print("\nPapers influenced by this one:")
for inf in result.influenced_papers:
    print(f"  <- {inf.title}  [influence={inf.influence_score:.3f}, sim={inf.similarity:.3f}]")
