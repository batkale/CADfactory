"""
Integration Test Cases for Semantic Search
Tests the ranking system with realistic CADfactory scenarios
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.semantic_search import (
    composite_ranking_score,
    detect_semantic_features,
)

# =============================================================================
# TEST CASES
# =============================================================================

TEST_SCENARIOS = [
    {
        "name": "Fidget Spinner Query",
        "query": "fidget spinner",
        "parts": [
            {
                "name": "Actual Fidget Spinner",
                "description": "Professional fidget spinner with 3 steel ball bearings, precision-engineered race, ergonomic ABS center grip, optimized for maximum spin duration and smooth rotation.",
                "manufacturing": "sla",
                "expected_rank": 1,
                "bom": ["608 ball bearing (qty 3)", "steel", "ABS plastic", "bearing lubrication"]
            },
            {
                "name": "Pop-it Toy",
                "description": "Pop-it silicone fidget toy with colorful bubble grid, satisfying tactile popping sensation, various shape patterns, stress relief for all ages.",
                "manufacturing": "injection",
                "expected_rank": 3,
                "bom": ["silicone rubber", "pigment"]
            },
            {
                "name": "Three-Arm Spinner",
                "description": "Lightweight three-arm spinner toy with balanced weight distribution, ergonomic center hold point, smooth rotation from momentum.",
                "manufacturing": "injection",
                "expected_rank": 2,
                "bom": ["ABS plastic", "weight inserts"]
            }
        ]
    },
    {
        "name": "Servo Bracket Query",
        "query": "servo mounting bracket",
        "parts": [
            {
                "name": "MG996R Servo Bracket",
                "description": "Servo bracket specifically designed for MG996R servo motors, includes 4x M3 bolt holes, aluminum construction for precision mounting, heat dissipation features.",
                "manufacturing": "cnc",
                "expected_rank": 1,
                "bom": ["aluminum 6061", "4x M3 bolts", "M3 washers"]
            },
            {
                "name": "Generic Phone Stand",
                "description": "Adjustable phone stand made from plastic, versatile bracket for various applications, not specifically designed for servo motors.",
                "manufacturing": "injection",
                "expected_rank": 2,
                "bom": ["ABS plastic"]
            }
        ]
    },
    {
        "name": "Enclosure Query",
        "query": "electronics enclosure",
        "parts": [
            {
                "name": "Weatherproof Outdoor Enclosure",
                "description": "IP67 weatherproof enclosure with rubber seals, mounting flanges, designed for outdoor electronics protection, cable glands included.",
                "manufacturing": "injection",
                "expected_rank": 1,
                "bom": ["ABS plastic", "rubber gasket", "stainless steel hardware"]
            },
            {
                "name": "Open Shelf Unit",
                "description": "Open storage shelving unit made from metal rods, not an enclosure, designed for open storage of items.",
                "manufacturing": "cnc",
                "expected_rank": 2,
                "bom": ["steel rod", "brackets"]
            }
        ]
    }
]


def run_test_scenario(scenario):
    """Run a single test scenario"""
    print(f"\n{'='*80}")
    print(f"TEST: {scenario['name']}")
    print(f"Query: '{scenario['query']}'")
    print(f"{'='*80}\n")

    query = scenario['query']
    parts = scenario['parts']

    scores = []

    # Score each part
    for part in parts:
        print(f"Part: {part['name']}")
        print(f"Description: {part['description'][:70]}...")

        # Detect features
        features = detect_semantic_features(part['description'], part.get('bom'))

        # Score
        score, breakdown = composite_ranking_score(
            query=query,
            description=part['description'],
            manufacturing_method=part['manufacturing'],
            features=features
        )

        print(f"Features detected: {[f.name for f in features]}")
        print(f"Score: {score:.1f}/100")
        print(f"Keyword match: {breakdown['keyword_match']:.1%}")
        print(f"Feature alignment: {breakdown['feature_alignment']:.1%}")
        print()

        scores.append({
            'name': part['name'],
            'expected_rank': part['expected_rank'],
            'score': score,
            'features': features,
            'breakdown': breakdown
        })

    # Sort and display ranking
    print("RANKING:")
    print("-" * 80)
    ranked = sorted(scores, key=lambda x: x['score'], reverse=True)

    test_passed = True
    for i, item in enumerate(ranked, 1):
        status = "✅" if i == item['expected_rank'] else "❌"
        print(f"{status} Rank #{i}: {item['name']} - {item['score']:.1f}/100 (expected rank: {item['expected_rank']})")
        if i != item['expected_rank']:
            test_passed = False

    print("-" * 80)
    result = "PASSED" if test_passed else "FAILED"
    print(f"Result: {result}")

    return test_passed


def main():
    print("\n" + "="*80)
    print("SEMANTIC SEARCH INTEGRATION TESTS")
    print("="*80)

    results = []
    for scenario in TEST_SCENARIOS:
        passed = run_test_scenario(scenario)
        results.append({
            'name': scenario['name'],
            'passed': passed
        })

    # Summary
    print("\n\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")

    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)

    for result in results:
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        print(f"{status}: {result['name']}")

    print(f"\nOverall: {passed_count}/{total_count} scenarios passed")

    if passed_count == total_count:
        print("\n🎉 All tests passed! The semantic search ranking is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Review the scoring logic.")

    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
