import os
import spacy
from scripts.models import model_ner
from core.config import settings

class Concept_NER:
    def __init__(self, ner_data: model_ner.NERModelRequest):
        if ner_data.concepts:
            self.concepts = ner_data.concepts
        
        self.project_directory_name = ner_data.project_directory_name
        self.spacy_model_name = "en_core_web_sm"
        self.ner_model_path = self._get_ner_model_path()
        
    ##Public Methods

    def train_concepts_ner_model(self):
        nlp = self._load_concepts_ner_model()
        if "entity_ruler" not in nlp.pipe_names:
            entity_ruler = nlp.add_pipe("entity_ruler", before="ner")
        else:
            entity_ruler = nlp.get_pipe("entity_ruler")
        
        patterns = [
            {"label": "CONCEPT", "pattern": concept.concept_name, "id": concept.concept_source} 
            for concept in self.concepts
        ]

        entity_ruler.add_patterns(patterns)
        nlp.to_disk(self.ner_model_path)

        #nlp = self._load_concepts_ner_model()
        ruler = nlp.get_pipe("entity_ruler")
        all_patterns = ruler.patterns
        print(len(all_patterns))
        #for pattern in all_patterns:
            #print(pattern)

    
    def fetch_concepts_with_ner_model(self, text):
        nlp = self._load_concepts_ner_model()
        doc = nlp(text)
        concepts_list = []
        for ent in doc.ents:
            if ent.label_ == "CONCEPT":
                concepts_list.append(model_ner.Concept(concept_name=ent.text, concept_source=ent.ent_id_))
        
        
        unique_concepts = {concept.concept_source: concept for concept in concepts_list}
        return list(unique_concepts.values())

    ##Private Methods

    def _load_concepts_ner_model(self):
        try:
            nlp = spacy.load(self.ner_model_path, enable=["entity_ruler"])
            return nlp
        except (OSError, FileNotFoundError):
            return self._load_spacy_model()
            #raise FileNotFoundError(f"Concept NER model does not exist at {self.ner_model_path}")

    def _get_ner_model_path(self): 
        wd = os.getcwd()
        project_directory_path = os.path.join(
            wd,
            settings.EVA_SETTINGS_PATH,
            self.project_directory_name,
            settings.EVA_SETTINGS_ENVIRONMENT_DIRECTORY
        )
        ner_model_name = f"{self.project_directory_name}_concept_ner_model"
        ner_model_path = os.path.join(project_directory_path, ner_model_name)
        return ner_model_path
    
    def _load_spacy_model(self):
        try:
            return spacy.load(self.spacy_model_name)
        except OSError:
            from spacy.cli import download
            download(self.spacy_model_name)
            return spacy.load(self.spacy_model_name)
