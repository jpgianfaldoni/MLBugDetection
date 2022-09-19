import pytest

def check_sanidade(model, examples, label):
    result = model.predict(examples.drop(label, axis=1))
    print(result)

check_sanidade()

@pytest.mark.parametrize()
def test_sanidade(a):
    assert a == 5