"""
Helix Global Corp - Advanced RAG Pipeline
Complete implementation with vector store, embeddings, and reasoning
"""

import json
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import os
from pathlib import Path


class SimpleEmbedding:
    """TF-IDF based embedding system"""
    
    def __init__(self):
        self.vocabulary = {}
        self.idf_scores = {}
        
    def fit(self, documents: List[str]):
        """Build vocabulary and IDF scores"""
        word_doc_count = defaultdict(int)
        all_words = set()
        
        for doc in documents:
            words = set(self._tokenize(doc))
            all_words.update(words)
            for word in words:
                word_doc_count[word] += 1
        
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        
        n_docs = len(documents)
        for word, count in word_doc_count.items():
            self.idf_scores[word] = np.log((n_docs + 1) / (count + 1)) + 1
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s\-_]', ' ', text)
        return [w for w in text.split() if len(w) > 2]
    
    def encode(self, text: str) -> np.ndarray:
        """Convert text to TF-IDF vector"""
        words = self._tokenize(text)
        vector = np.zeros(len(self.vocabulary))
        
        word_count = defaultdict(int)
        for word in words:
            word_count[word] += 1
        
        for word, count in word_count.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                tf = count / max(len(words), 1)
                idf = self.idf_scores.get(word, 1.0)
                vector[idx] = tf * idf
        
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector


class VectorStore:
    """In-memory vector store"""
    
    def __init__(self, embedding_model: SimpleEmbedding):
        self.embedding_model = embedding_model
        self.vectors = []
        self.metadata = []
        
    def add(self, texts: List[str], metadatas: List[Dict]):
        """Add documents"""
        for text, meta in zip(texts, metadatas):
            vector = self.embedding_model.encode(text)
            self.vectors.append(vector)
            self.metadata.append(meta)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, Dict, float]]:
        """Search similar documents"""
        query_vector = self.embedding_model.encode(query)
        
        similarities = []
        for idx, vec in enumerate(self.vectors):
            sim = np.dot(query_vector, vec)
            similarities.append((idx, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, sim in similarities[:k]:
            meta = self.metadata[idx]
            text = meta.get('text', '')
            results.append((text, meta, float(sim)))
        
        return results


class PolicyExtractor:
    """Extract policy rules from text"""
    
    def __init__(self, policy_text: str):
        self.text = policy_text
        
    def extract_rules(self) -> List[Dict]:
        """Extract specific rules"""
        rules = []
        
        # Leave entitlements
        leave_matches = re.finditer(r'(\d+)\s+days?\s+of\s+(?:paid\s+)?(\w+(?:\s+\w+)?)\s+leave', 
                                    self.text, re.IGNORECASE)
        for match in leave_matches:
            rules.append({
                'type': 'leave_entitlement',
                'days': int(match.group(1)),
                'leave_type': match.group(2),
                'source': 'policy_document'
            })
        
        # Tenure benefits
        tenure_matches = re.finditer(r'(\d+)(?:\+)?\s+years.*?(\d+)\s+days', 
                                     self.text, re.IGNORECASE)
        for match in tenure_matches:
            rules.append({
                'type': 'tenure_benefit',
                'years': int(match.group(1)),
                'days': int(match.group(2)),
                'source': 'policy_document'
            })
        
        return rules


class DataIntegrityChecker:
    """Data quality validation"""
    
    def check_employee_data(self, emp_df: pd.DataFrame) -> Dict:
        """Check employee data"""
        issues = []
        
        # Duplicates
        duplicates = emp_df[emp_df.duplicated(subset=['emp_id'], keep=False)]
        if not duplicates.empty:
            issues.append({
                'type': 'duplicate_employees',
                'count': len(duplicates)
            })
        
        # Missing values
        missing = emp_df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                issues.append({
                    'type': 'missing_values',
                    'column': col,
                    'count': int(count)
                })
        
        quality_score = max(0, 100 - (len(issues) / max(len(emp_df), 1) * 100))
        
        return {
            'total_records': len(emp_df),
            'issues': issues,
            'quality_score': round(quality_score, 2)
        }
    
    def check_leave_data(self, leave_df: pd.DataFrame) -> Dict:
        """Check leave data"""
        issues = []
        
        # Negative days
        negative = leave_df[leave_df['days'] < 0]
        if not negative.empty:
            issues.append({
                'type': 'negative_days',
                'count': len(negative)
            })
        
        quality_score = max(0, 100 - (len(issues) / max(len(leave_df), 1) * 100))
        
        return {
            'total_records': len(leave_df),
            'issues': issues,
            'quality_score': round(quality_score, 2)
        }


class LogicalReasoner:
    """Business logic processor"""
    
    def __init__(self, emp_df: pd.DataFrame, leave_df: pd.DataFrame, balance_df: pd.DataFrame):
        self.emp_df = emp_df
        self.leave_df = leave_df
        self.balance_df = balance_df
        
    def calculate_tenure_benefits(self, emp_id: str) -> Dict:
        """Calculate tenure-based benefits"""
        emp = self.emp_df[self.emp_df['emp_id'] == emp_id]
        
        if emp.empty:
            return {'error': 'Employee not found'}
        
        joining_date = pd.to_datetime(emp.iloc[0]['joining_date'])
        today = pd.Timestamp.now()
        tenure_years = (today - joining_date).days / 365.25
        
        annual_leave = 15
        if tenure_years >= 5:
            annual_leave = 20
            tier = "TIER 2"
        elif tenure_years >= 3:
            annual_leave = 17
            tier = "TIER 1"
        else:
            tier = "Base"
        
        location = emp.iloc[0]['location']
        location_bonus = 8 if location == 'London' else 0
        
        return {
            'emp_id': emp_id,
            'name': emp.iloc[0].get('name', 'Unknown'),
            'tenure_years': round(tenure_years, 2),
            'tier': tier,
            'base_annual': 15,
            'tenure_bonus': annual_leave - 15,
            'location_bonus': location_bonus,
            'total_entitlement': annual_leave + location_bonus,
            'sick_leave': 10,
            'emergency_leave': 3
        }
    
    def check_leave_eligibility(self, emp_id: str, leave_type: str, days: int) -> Dict:
        """Check leave eligibility"""
        balance = self.balance_df[self.balance_df['emp_id'] == emp_id]
        
        if balance.empty:
            return {'eligible': False, 'reason': 'Balance not found'}
        
        type_mapping = {
            'annual': 'annual_bal',
            'sick': 'sick_bal',
            'loyalty': 'loyalty_bal',
            'emergency': 'emergency_bal'
        }
        
        balance_col = type_mapping.get(leave_type.lower())
        if not balance_col:
            return {'eligible': False, 'reason': f'Unknown leave type: {leave_type}'}
        
        available = balance.iloc[0][balance_col]
        
        if days > available:
            return {
                'eligible': False,
                'reason': f'Insufficient balance. Available: {available}, Requested: {days}',
                'available': int(available),
                'requested': days
            }
        
        return {
            'eligible': True,
            'available': int(available),
            'requested': days,
            'remaining': int(available - days)
        }
    
    def check_singapore_mc(self, emp_id: str) -> Dict:
        """Check Singapore MC requirement"""
        emp = self.emp_df[self.emp_df['emp_id'] == emp_id]
        
        if emp.empty:
            return {'requires_mc': False}
        
        location = emp.iloc[0]['location']
        
        return {
            'requires_mc': location == 'Singapore',
            'location': location,
            'policy': 'MC required for ALL sick leave' if location == 'Singapore' else 'MC required after 2 days'
        }


class HallucinationDetector:
    """Detect hallucinations in responses"""
    
    def __init__(self, source_docs: List[str]):
        self.source_text = ' '.join(source_docs).lower()
        
    def verify_claim(self, claim: str) -> Dict:
        """Verify claim against sources"""
        claim_lower = claim.lower()
        numbers = re.findall(r'\d+', claim)
        
        verified_numbers = [n for n in numbers if n in self.source_text]
        unverified_numbers = [n for n in numbers if n not in self.source_text]
        
        # Check text overlap
        claim_words = set(claim_lower.split())
        source_words = set(self.source_text.split())
        overlap = len(claim_words & source_words) / max(len(claim_words), 1)
        
        verified = len(unverified_numbers) == 0 and overlap > 0.3
        confidence = 0.9 if verified else (0.5 if overlap > 0.2 else 0.3)
        
        return {
            'verified': verified,
            'confidence': round(confidence, 2),
            'overlap_score': round(overlap, 2)
        }


class RAGPipeline:
    """Main RAG orchestrator"""
    
    def __init__(self):
        self.embedding_model = SimpleEmbedding()
        self.vector_store = VectorStore(self.embedding_model)
        self.integrity_checker = DataIntegrityChecker()
        self.data_loaded = False
        
    def load_data(self, data_dir: str, policy_text: str):
        """Load all data sources"""
        # Load Excel
        excel_path = f"{data_dir}/leave_intelligence.xlsx"
        self.leave_history = pd.read_excel(excel_path, sheet_name='Leave_History')
        self.balances = pd.read_excel(excel_path, sheet_name='Available_Balances')
        self.dept_analytics = pd.read_excel(excel_path, sheet_name='Dept_Analytics')
        
        # Load or create employee data
        emp_path = f"{data_dir}/employee_master.csv"
        if os.path.exists(emp_path):
            self.employees = pd.read_csv(emp_path)
        else:
            # Create sample data
            self.employees = self._create_sample_employees()
        
        self.policy_text = policy_text
        self.data_loaded = True
        
        return {
            'employees': len(self.employees),
            'leave_records': len(self.leave_history),
            'balances': len(self.balances)
        }
    
    def _create_sample_employees(self) -> pd.DataFrame:
        """Create sample employee data"""
        locations = ['Singapore', 'London', 'New York', 'Bangalore', 'Tokyo']
        departments = ['Engineering', 'Product', 'Marketing', 'Finance', 'HR', 'Legal', 'Operations', 'IT', 'Customer Success']
        
        data = []
        for i in range(500):
            emp_id = f'EMP{1001+i}'
            data.append({
                'emp_id': emp_id,
                'name': f'Employee {i+1}',
                'department': departments[i % len(departments)],
                'location': locations[i % len(locations)],
                'joining_date': (pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(365, 3650))).strftime('%Y-%m-%d'),
                'salary_band': chr(65 + (i % 5)),  # A-E
                'is_active': True
            })
        
        return pd.DataFrame(data)
    
    def build_index(self):
        """Build vector index"""
        if not self.data_loaded:
            raise Exception("Data not loaded")
        
        documents = []
        metadatas = []
        
        # Index policy
        extractor = PolicyExtractor(self.policy_text)
        chunks = self._chunk_text(self.policy_text, 400)
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                'source': 'policy',
                'chunk_id': i,
                'text': chunk
            })
        
        # Index employee samples
        for _, emp in self.employees.head(100).iterrows():
            text = f"{emp['emp_id']} {emp['name']} - {emp['department']} dept, {emp['location']}, joined {emp['joining_date']}"
            documents.append(text)
            metadatas.append({
                'source': 'employee',
                'emp_id': emp['emp_id'],
                'text': text
            })
        
        # Index leave stats
        for leave_type in self.leave_history['leave_type'].unique():
            subset = self.leave_history[self.leave_history['leave_type'] == leave_type]
            text = f"{leave_type} leave: {len(subset)} applications, average {subset['days'].mean():.1f} days"
            documents.append(text)
            metadatas.append({
                'source': 'leave_stats',
                'leave_type': leave_type,
                'text': text
            })
        
        self.embedding_model.fit(documents)
        self.vector_store.add(documents, metadatas)
        
        return len(documents)
    
    def _chunk_text(self, text: str, chunk_size: int = 400) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - 50):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """Query the RAG system"""
        results = self.vector_store.similarity_search(question, k=top_k)
        
        context = '\n\n'.join([text for text, _, _ in results])
        sources = [meta for _, meta, _ in results]
        
        return {
            'question': question,
            'context': context,
            'sources': sources,
            'num_sources': len(results)
        }
    
    def analyze_integrity(self) -> Dict:
        """Analyze data integrity"""
        emp_check = self.integrity_checker.check_employee_data(self.employees)
        leave_check = self.integrity_checker.check_leave_data(self.leave_history)
        
        return {
            'employee_data': emp_check,
            'leave_data': leave_check,
            'overall_score': round((emp_check['quality_score'] + leave_check['quality_score']) / 2, 2)
        }
    
    def get_reasoner(self) -> LogicalReasoner:
        """Get logical reasoner instance"""
        return LogicalReasoner(self.employees, self.leave_history, self.balances)
