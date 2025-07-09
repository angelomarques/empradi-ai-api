from datetime import datetime
from typing import Optional
from bson import ObjectId
from flask_pymongo import PyMongo


class Prompt:
    def __init__(
        self,
        name: str,
        content: str,
        description: str = "",
        is_active: bool = True,
        version: str = "1.0",
    ):
        self.name = name
        self.content = content
        self.description = description
        self.is_active = is_active
        self.version = version
        self.created_at = datetime.today()
        self.updated_at = datetime.today()

    def to_dict(self):
        return {
            "name": self.name,
            "content": self.content,
            "description": self.description,
            "is_active": self.is_active,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(data: dict):
        prompt = Prompt(
            name=data["name"],
            content=data["content"],
            description=data.get("description", ""),
            is_active=data.get("is_active", True),
            version=data.get("version", "1.0"),
        )
        prompt.created_at = data.get("created_at", datetime.today())
        prompt.updated_at = data.get("updated_at", datetime.today())
        return prompt


class PromptModel:
    def __init__(self, mongo: PyMongo):
        if mongo is None:
            raise ValueError("MongoDB connection is not properly initialized")
        self.mongo = mongo
        self.collection = mongo.db.emprad_prompts

    def create(self, prompt: Prompt) -> str:
        """Create a new prompt in the database."""
        if self.collection is None:
            raise ValueError("Prompts collection is not properly initialized")
        result = self.collection.insert_one(prompt.to_dict())
        return str(result.inserted_id)

    def get_by_id(self, prompt_id: str) -> Optional[Prompt]:
        """Get a prompt by its ID."""
        if self.collection is None:
            raise ValueError("Prompts collection is not properly initialized")
        result = self.collection.find_one({"_id": ObjectId(prompt_id)})
        if result:
            return Prompt.from_dict(result)
        return None

    def get_by_name(self, name: str) -> Optional[Prompt]:
        """Get a prompt by its name."""
        if self.collection is None:
            raise ValueError("Prompts collection is not properly initialized")
        result = self.collection.find_one({"name": name, "is_active": True})
        if result:
            return Prompt.from_dict(result)
        return None

    def get_active_prompt(self) -> Optional[Prompt]:
        """Get the currently active prompt."""
        if self.collection is None:
            raise ValueError("Prompts collection is not properly initialized")
        result = self.collection.find_one({"is_active": True})
        if result:
            return Prompt.from_dict(result)
        return None

    def get_all(self) -> list[Prompt]:
        """Get all prompts."""
        if self.collection is None:
            raise ValueError("Prompts collection is not properly initialized")
        results = self.collection.find().sort("created_at", -1)
        return [Prompt.from_dict(result) for result in results]

    def update(self, prompt_id: str, prompt: Prompt) -> bool:
        """Update a prompt."""
        if self.collection is None:
            raise ValueError("Prompts collection is not properly initialized")
        prompt_dict = prompt.to_dict()
        prompt_dict["updated_at"] = datetime.today()
        result = self.collection.update_one(
            {"_id": ObjectId(prompt_id)}, {"$set": prompt_dict}
        )
        return result.modified_count > 0

    def delete(self, prompt_id: str) -> bool:
        """Delete a prompt."""
        if self.collection is None:
            raise ValueError("Prompts collection is not properly initialized")
        result = self.collection.delete_one({"_id": ObjectId(prompt_id)})
        return result.deleted_count > 0

    def set_active(self, prompt_id: str) -> bool:
        """Set a prompt as active and deactivate all others."""
        if self.collection is None:
            raise ValueError("Prompts collection is not properly initialized")

        # First, deactivate all prompts
        self.collection.update_many({}, {"$set": {"is_active": False}})

        # Then, activate the specified prompt
        result = self.collection.update_one(
            {"_id": ObjectId(prompt_id)},
            {"$set": {"is_active": True, "updated_at": datetime.today()}},
        )
        return result.modified_count > 0

    def get_default_prompt(self) -> str:
        """Get the default prompt content or return a fallback."""
        active_prompt = self.get_active_prompt()
        if active_prompt:
            return active_prompt.content

        # Fallback prompt if no active prompt is found
        return """### 🧠 Prompt para o Assistente de IA – EMPRAD 2025 (Formato OpenAI)

Você é o **Assistente EMPRAD**, uma inteligência artificial treinada para responder exclusivamente com base nos **artigos publicados no EMPRAD 2025 (Encontro de Empreendedorismo e Gestão para o Desenvolvimento)**.

Seu objetivo é **ajudar participantes do evento a encontrarem artigos relevantes** sobre temas de empreendedorismo, administração e negócios, trazendo respostas fundamentadas, confiáveis e referenciadas.

---

### 🎯 Regras de Comportamento

1. **Base de conhecimento restrita**:
    
    Você só pode responder com base no conteúdo dos artigos publicados no EMPRAD.
    
    - **Não use conhecimento externo** ou invente informações.
    - Se a base não contém resposta para a pergunta, diga:
        
        > "Não encontrei artigos no EMPRAD 2025 que abordem diretamente essa questão."
        > 
2. **Temas obrigatórios**:
    
    Aceite apenas perguntas relacionadas a:
    
    - Empreendedorismo
    - Startups
    - Inovação
    - Gestão e administração
    - Estratégia organizacional
    - Negócios de impacto
    - Sustentabilidade empresarial
    - Políticas públicas voltadas ao desenvolvimento
    - Finanças e investimentos em novos negócios
    - Educação empreendedora
    
    Se o tema estiver fora desse escopo, responda:
    
    > "Este assistente é voltado apenas a temas abordados no EMPRAD. Reformule sua pergunta com foco em empreendedorismo, negócios ou administração."
    > 
3. **Linguagem inadequada**:
    
    Não aceite perguntas com palavrões, ofensas ou termos depreciativos.
    
    Se detectar esse tipo de linguagem, responda:
    
    > "Sua pergunta contém termos inadequados. Reformule-a de forma respeitosa e dentro do escopo temático do evento."
    > 

---

### 🖼️ Formato da Resposta (Obrigatório)

Toda resposta deve seguir a estrutura abaixo:

1. **Resposta objetiva e resumida**:
    - Comece com uma breve explicação com base nos artigos encontrados.
    - Destaque as principais descobertas, abordagens metodológicas e implicações práticas observadas nos estudos.

---

### ✅ Exemplo de resposta ideal

> Com base na sua consulta sobre "O que é uma startup?", analisei os anais do EMPRAD e encontrei 3 artigos relevantes publicados entre 2023 e 2025.
> 
> 
> **Síntese das descobertas**: Os estudos apontam para a importância da contextualização das práticas de gestão e empreendedorismo à realidade brasileira, considerando as especificidades culturais, econômicas e sociais dos países em desenvolvimento.
> 
> **Abordagens metodológicas**: Análises comparativas entre diferentes regiões e setores, evidenciando padrões distintos entre startups urbanas e rurais.
> 
> **Implicações práticas**: A literatura recomenda que startups adotem práticas colaborativas e desenvolvam visão sistêmica para enfrentar ambientes de negócios dinâmicos e incertos.
>"""
