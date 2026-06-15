from app.embeddings import get_model
import numpy as np

def test_get_model_singleton_behavior():
    # Calling get_model repeatedly should work
    model1 = get_model()
    model2 = get_model()
    assert hasattr(model1, "encode")
    assert hasattr(model2, "encode")
    
def test_model_encode_behavior():
    model = get_model()
    vector = model.encode("Test string")
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0
