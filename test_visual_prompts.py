"""
Test du système de génération de prompts intelligents
"""

from backend_multimodal import generate_visual_prompt_from_story

# Test 1: AWD System
story_awd = """The all-wheel drive system is an advanced automotive technology that intelligently distributes engine torque between the front and rear axles to optimize traction and handling.

When the vehicle accelerates, sensors continuously monitor wheel speed and traction conditions. The system uses a center differential or electronic coupling to split power between the axles."""

query_awd = "Explain how the all-wheel drive system distributes torque"

print("=" * 80)
print("TEST 1: AWD SYSTEM")
print("=" * 80)
print(f"User Query: {query_awd}")
print(f"\nStory Excerpt: {story_awd[:150]}...")
print(f"\n📸 Generated Visual Prompt:")
print(generate_visual_prompt_from_story(story_awd, query_awd))

# Test 2: Turbo Engine
story_turbo = """The turbocharged engine uses exhaust gases to spin a turbine that compresses incoming air, forcing more oxygen into the combustion chamber. This results in significantly more power without increasing engine size."""

query_turbo = "How does a turbocharger work?"

print("\n" + "=" * 80)
print("TEST 2: TURBOCHARGER")
print("=" * 80)
print(f"User Query: {query_turbo}")
print(f"\nStory Excerpt: {story_turbo[:150]}...")
print(f"\n📸 Generated Visual Prompt:")
print(generate_visual_prompt_from_story(story_turbo, query_turbo))

# Test 3: Electric Vehicle
story_ev = """The electric vehicle powertrain consists of a high-voltage battery pack that stores electrical energy, which is then converted by an electric motor into mechanical power to drive the wheels."""

query_ev = "Explain electric vehicle powertrain"

print("\n" + "=" * 80)
print("TEST 3: ELECTRIC VEHICLE")
print("=" * 80)
print(f"User Query: {query_ev}")
print(f"\nStory Excerpt: {story_ev[:150]}...")
print(f"\n📸 Generated Visual Prompt:")
print(generate_visual_prompt_from_story(story_ev, query_ev))

# Test 4: Suspension
story_suspension = """The automotive suspension system absorbs road irregularities using shock absorbers and springs, maintaining tire contact with the road while providing passenger comfort."""

query_suspension = "How does car suspension work?"

print("\n" + "=" * 80)
print("TEST 4: SUSPENSION SYSTEM")
print("=" * 80)
print(f"User Query: {query_suspension}")
print(f"\nStory Excerpt: {story_suspension[:150]}...")
print(f"\n📸 Generated Visual Prompt:")
print(generate_visual_prompt_from_story(story_suspension, query_suspension))

print("\n" + "=" * 80)
print("✅ TESTS TERMINÉS")
print("=" * 80)
