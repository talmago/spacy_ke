import pytest
import spacy

from packaging.version import Version


@pytest.fixture(scope="session", autouse=True)
def en_core_web_sm():
    return spacy.load("en_core_web_sm")


@pytest.fixture(scope="session", autouse=True)
def spacy_v3():
    return Version(spacy.__version__) >= Version("3.0")
