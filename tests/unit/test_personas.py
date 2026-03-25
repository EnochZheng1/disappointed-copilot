"""Tests for persona definitions."""

from disappointed.commentary.personas import PERSONAS, get_persona, get_roast_prompt


def test_all_personas_exist():
    assert "british_instructor" in PERSONAS
    assert "tired_mom" in PERSONAS
    assert "deadpan_ai" in PERSONAS


def test_get_persona_returns_correct():
    persona = get_persona("deadpan_ai")
    assert persona.name == "deadpan_ai"
    assert "DAVE-9000" in persona.display_name


def test_get_persona_falls_back():
    persona = get_persona("nonexistent_pack")
    assert persona.name == "british_instructor"


def test_roast_prompt_includes_context():
    persona = get_persona("british_instructor")
    prompt = get_roast_prompt(persona, "tailgater", "A BMW is 2 feet behind us")
    assert "BMW" in prompt
    assert "tailgater" not in prompt or "example" in prompt.lower()


def test_all_personas_have_all_trigger_examples():
    trigger_types = ["tailgater", "swerver", "green_light", "self_critique", "hard_brake"]
    for name, persona in PERSONAS.items():
        for trigger in trigger_types:
            assert trigger in persona.example_roasts, f"{name} missing example for {trigger}"
