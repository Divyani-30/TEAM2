"""
Ollama Integration for Mistral 7B Instruct
Handles LLM query processing with RAG context
"""

import subprocess
import json
import re
from typing import Dict, Optional


class OllamaLLM:
    """Ollama Mistral integration"""
    
    def __init__(self, model: str = "mistral:7b-instruct"):
        self.model = model
        
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.3) -> str:
        """Generate response"""
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            result = subprocess.run(
                ['ollama', 'run', self.model, full_prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            return f"Error: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            return "Error: Timeout"
        except FileNotFoundError:
            return "Error: Ollama not installed. Using fallback response."
        except Exception as e:
            return f"Error: {str(e)}"


class PromptBuilder:
    """Build prompts for different query types"""
    
    SYSTEM_PROMPT = """You are an HR Intelligence Assistant for Helix Global Corp.

Your role:
1. Provide accurate information from company policies and data
2. Apply business logic correctly
3. Cite sources using [1], [2] format
4. Acknowledge uncertainty
5. Never hallucinate

Be concise, professional, and cite specific policy sections."""
    
    @staticmethod
    def build_rag_prompt(question: str, context: str, sources: list) -> str:
        """Build RAG query prompt"""
        sources_text = "\n".join([
            f"[{i+1}] {src.get('source', 'unknown')}"
            for i, src in enumerate(sources)
        ])
        
        return f"""Answer based on this context:

CONTEXT:
{context}

SOURCES:
{sources_text}

QUESTION: {question}

Provide a clear answer with [source] citations. If uncertain, say so.

ANSWER:"""
    
    @staticmethod
    def build_calculation_prompt(question: str, data: Dict) -> str:
        """Build calculation prompt"""
        return f"""Calculate based on this data:

DATA:
{json.dumps(data, indent=2)}

QUESTION: {question}

Show step-by-step calculation and final answer.

CALCULATION:"""


class QueryProcessor:
    """Process queries through RAG + LLM"""
    
    def __init__(self, rag_pipeline, llm: OllamaLLM):
        self.rag = rag_pipeline
        self.llm = llm
        self.prompt_builder = PromptBuilder()
        
    def process(self, question: str, query_type: str = "general") -> Dict:
        """Process query"""
        # Retrieve context
        rag_result = self.rag.query(question)
        
        # Build prompt
        prompt = self.prompt_builder.build_rag_prompt(
            question, 
            rag_result['context'], 
            rag_result['sources']
        )
        
        # Generate response
        response = self.llm.generate(
            prompt,
            system_prompt=self.prompt_builder.SYSTEM_PROMPT
        )
        
        # Verify
        from rag_system import HallucinationDetector
        detector = HallucinationDetector([rag_result['context']])
        
        sentences = re.split(r'[.!?]', response)[:3]
        verifications = [
            detector.verify_claim(s.strip()) 
            for s in sentences if len(s.strip()) > 10
        ]
        
        avg_confidence = sum(v['confidence'] for v in verifications) / max(len(verifications), 1)
        
        return {
            'question': question,
            'answer': response,
            'sources': rag_result['sources'],
            'confidence': round(avg_confidence, 2),
            'verification': verifications
        }
    
    def process_employee_query(self, emp_id: str, question: str) -> Dict:
        """Process employee-specific query"""
        reasoner = self.rag.get_reasoner()
        
        # Get employee data
        tenure_info = reasoner.calculate_tenure_benefits(emp_id)
        
        if 'error' in tenure_info:
            return {'error': tenure_info['error']}
        
        # Build prompt
        prompt = self.prompt_builder.build_calculation_prompt(question, tenure_info)
        response = self.llm.generate(prompt)
        
        return {
            'emp_id': emp_id,
            'question': question,
            'employee_data': tenure_info,
            'answer': response
        }
