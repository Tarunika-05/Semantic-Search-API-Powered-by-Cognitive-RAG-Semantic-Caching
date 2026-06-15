from app.dataset import load_documents

def test_load_documents_coverage(monkeypatch):
    # Mock load_dataset to return a tiny fake dataset
    class MockDataset:
        def select(self, r):
            return [{"title": "Fake Title", "abstract": "Fake abstract with vision and image keywords so it gets cs.CV."}]
            
    def mock_load_dataset(*args, **kwargs):
        return MockDataset()
        
    monkeypatch.setattr("app.dataset.load_dataset", mock_load_dataset)
    
    docs, labels, label_names = load_documents()
    assert len(docs) == 1
    assert "Fake Title" in docs[0]
    assert len(labels) == 1
    assert "cs.CV" in label_names
