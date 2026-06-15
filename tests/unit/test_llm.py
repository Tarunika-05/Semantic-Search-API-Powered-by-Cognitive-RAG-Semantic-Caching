from app.llm import generate_answer, DummyProvider

def test_dummy_provider():
    provider = DummyProvider()
    prompt = "This is a prompt with docs: ML is cool, ML uses data"
    
    result = provider.generate(prompt)
    assert "[SIMULATED AI ANSWER]" in result

def test_generate_answer_dummy():
    provider = DummyProvider()
    query = "What is machine learning?"
    docs = ["Doc 1: ML is cool", "Doc 2: ML uses data"]
    
    result = generate_answer(query, docs, provider=provider)
    assert "[SIMULATED AI ANSWER]" in result

def test_generate_answer_empty_docs():
    provider = DummyProvider()
    # Should handle empty docs safely
    result = generate_answer("query", [], provider=provider)
    assert result is None or isinstance(result, str)
